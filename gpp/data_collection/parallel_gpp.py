import os
import json
import torch
import sys
from pathlib import Path
from torch.utils.data import Subset
import math


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpp.dataset.data_utils import load_data_normal, load_data_folder
from gpp.utils.visual_utils import patchify, viz_patches

from gpp.genetic_algo.config import load_config, resolve_device, default_config_path
from gpp.model.clip_model import load_clip
from gpp.genetic_algo.runner import patch_modified_clip, parallel_patch_modified_clip
import time


# ─── Load Config ─────────────────────────────────────────────────────
# cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
cfg_path = '/home/atuin/v123be/v123be15/GeneticPatchPruning/config/p_data_collection.yaml'

print(f'{cfg_path = }')
cfg = load_config(cfg_path)
print(f'{cfg = }')
device = cfg["device"]
# device = resolve_device(device)
task = cfg["task"]
# print(f'{task = }')
# ─── Load Data Config ────────────────────────────────────────────────
save_path = cfg.get("save_path", "./data/")
num_samples = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_dir = cfg["data_dir"]
# print(f'{data_dir = }')
data_split = cfg['split']
# print(f'{data_split = }')
cache_dir = cfg['cache_dir']
# print(f'{cache_dir = }')
# ─── Load GA config ────────────────────────────────────────────────
keep_pct = cfg["keep_pct"]
viz = cfg["visualize"]
optimize_keep = cfg.get("optimize_keep", False)
min_keep_pct = cfg.get("min_keep_pct", 0.1)
max_keep_pct = cfg.get("max_keep_pct", 0.9)
keep_penalty = cfg.get("keep_penalty", 0.1)
# ml_model = cfg["model_checkpoint"]
# ─── Load Model ──────────────────────────────────────────────────────
model_id = cfg["model_id"] # clip model id




def get_dist_info():
    # Priority: torchrun envs, then Slurm, then single-process fallback
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = 0

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    else:
        rank = local_rank

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_GPUS_ON_NODE", 1)),
        )
    )
    print(f'{rank = }, {world_size = }, {local_rank = }')
    return rank, world_size, local_rank


def slice_data(dataset, prompts, start, end):
    # Dataset can be list-like or torch Dataset
    if isinstance(dataset, list):
        dataset = dataset[start:end]
    else:
        dataset = Subset(dataset, list(range(start, end)))

    if isinstance(prompts, (list, tuple)):
        prompts = prompts[start:end]
    return dataset, prompts


# ─── Main Function ───────────────────────────────────────────────────

def main():
    rank, world_size, local_rank = get_dist_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    print(
        f"Rank {rank}/{world_size} using device {device} "
        f"for {dataset_name} (num_samples={num_samples})"
    )

    print(f'Loading and Using model: {model_id} on {device}...')

    model, processor = load_clip(model_id, device)

    # Load full dataset or requested total
    dataset, prompts = load_data_folder(
        data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir
    )
    print(f"loaded dataset with {len(prompts)} samples.")

    total_samples = num_samples
    if total_samples == 0:
        total_samples = len(dataset)

    per_rank = math.ceil(total_samples / world_size)
    start = rank * per_rank
    end = min(total_samples, start + per_rank)

    if start >= total_samples:
        print(f"Rank {rank} has no samples to process. Exiting.")
        return

    dataset, _ = slice_data(dataset, prompts, start, end)
    print(f"Rank {rank} processing prompts{len(prompts)} samples.")
    print(f"Rank {rank} processing samples [{start}, {end-1}]")

    # Load model on this rank’s device
    model, processor = load_clip(model_id, device)

    # Use per-rank output to avoid write contention
    out_dir = os.path.join(save_path, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path_jsonl = os.path.join(
        out_dir, f"{start}_{end-1}_{keep_penalty}.jsonl"
    )

    _ = parallel_patch_modified_clip(
        dataset,
        prompts,
        model,
        processor,
        device,
        keep_pct,
        out_path_jsonl,
        viz=viz,
        patchify_fn=patchify,
        viz_patches_fn=viz_patches,
        optimize_keep=optimize_keep,
        min_keep_pct=min_keep_pct,
        max_keep_pct=max_keep_pct,
        keep_penalty=keep_penalty,
        base_idx=start,
    )


    print(
        f"Rank {rank}: saved results to {out_path_jsonl} "
        f"(resume supported)."
    )

# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

