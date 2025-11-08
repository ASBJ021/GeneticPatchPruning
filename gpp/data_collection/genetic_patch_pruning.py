import os
import json
import torch
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpp.dataset.data_utils import load_data_normal
from gpp.utils.visual_utils import patchify, viz_patches

from gpp.genetic_algo.config import load_config, resolve_device, default_config_path
from gpp.model.clip_model import load_clip
from gpp.genetic_algo.runner import patch_modified_clip
import time


# ─── Load Config ─────────────────────────────────────────────────────
# cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
cfg_path = '/var/lit2425/jenga/GeneticPatchPruning/config/data_colletion.yaml'

print(f'{cfg_path = }')
cfg = load_config(cfg_path)
print(f'{cfg = }')
device = resolve_device(cfg)

task = cfg["task"]
save_path = cfg.get("save_path", "./data/")
num_samples = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_split = cfg['split']
model_id = cfg["model_id"]
keep_pct = cfg["keep_pct"]
viz = cfg["visualize"]
optimize_keep = cfg.get("optimize_keep", False)
min_keep_pct = cfg.get("min_keep_pct", 0.1)
max_keep_pct = cfg.get("max_keep_pct", 0.9)
keep_penalty = cfg.get("keep_penalty", 0.1)
# ml_model = cfg["model_checkpoint"]
# ─── Load Model ──────────────────────────────────────────────────────
model, processor = load_clip(model_id, device)


# ─── Main Function ───────────────────────────────────────────────────
def main():

    print(
        f"Creating data from {'full' if num_samples == 0 else num_samples} samples of {dataset_name} dataset"
    )

    dataset, prompts = load_data_normal(dataset_name, num_samples, SPLIT=data_split)
    # print(f'{prompts =}')

    # out_path_jsonl = f"{dataset_name}_{num_samples}_final_patches_{int(keep_penalty * 100)}.jsonl"
    # Preserve original behavior: override path with hard-coded target
    out_path_jsonl = (
        f"{save_path}{dataset_name}/{num_samples}_{keep_penalty}_{time.time()}.jsonl"
    )

    # out_path_jsonl = '/home/utn/firi22ka/Desktop/jenga/Adaptive-Tokenization/new_src/clane9/imagenet-100_500_0.5_1758265212.1184783.jsonl'

    _ = patch_modified_clip(
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
    )

    print(
        f"Per-image results saved line-by-line to {out_path_jsonl} (resume supported)."
    )


# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
