import os
import sys
import time
from pathlib import Path
from typing import Optional

import clip
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpp.dataset.data_utils import load_data_folder, load_data_normal
from gpp.eval.compare import display_comparison_tables
from gpp.genetic_algo.io import save_record_jsonl
from gpp.model.clip_model import forward_with_selected_patches
from gpp.model.model import (
    LightweightPatchSelector,
    PatchSelector,
    PatchSelectorResBlock,
    PatchSelectorWithSoftmax,
    SimplePatchSelector,
    SimplePatchSelectorWithDropout,
)


def build_selector(mlp: str, device: str):
    outputs_are_probs = False
    if mlp == "PatchSelector":
        selector = PatchSelector().to(device)
    elif mlp == "PatchSelectorWithSoftmax":
        selector = PatchSelectorWithSoftmax().to(device)
        outputs_are_probs = True
    elif mlp == "SimplePatchSelector":
        selector = SimplePatchSelector().to(device)
    elif mlp == "SimplePatchSelectorWithDropout":
        selector = SimplePatchSelectorWithDropout().to(device)
    elif mlp == "PatchSelectorResBlock":
        selector = PatchSelectorResBlock().to(device)
    elif mlp == "LightweightPatchSelector":
        selector = LightweightPatchSelector().to(device)
    else:
        raise ValueError(f"Unsupported mlp: {mlp}")
    return selector, outputs_are_probs


@torch.no_grad()
def get_text_features(prompts, model, device: str) -> torch.Tensor:
    text_tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def predict_with_selector(
    img,
    selector,
    clip_model,
    clip_proc,
    text_features: torch.Tensor,
    device: str,
    outputs_are_probs: bool,
    pred_thresh: float,
    min_keep_tokens: int,
):
    pixel_values = clip_proc(img).unsqueeze(0).to(device)

    x = clip_model.visual.conv1(pixel_values)
    bsz, dim, _, _ = x.shape
    patch_preln = x.reshape(bsz, dim, -1).permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(bsz, -1, -1)
    tokens = torch.cat([cls_token, patch_preln], dim=1)
    tokens = tokens + clip_model.visual.positional_embedding.unsqueeze(0)
    tokens = clip_model.visual.ln_pre(tokens)
    tokens = clip_model.visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)

    patch_tokens = tokens[:, 1:, :]
    raw = selector(patch_tokens).squeeze(-1)
    probs = raw if outputs_are_probs else torch.sigmoid(raw)
    patch_probs = probs[0]

    selected_mask = patch_probs >= pred_thresh
    n_tokens = int(patch_probs.numel())
    min_keep = max(1, min(int(min_keep_tokens), n_tokens))

    if int(selected_mask.sum().item()) < min_keep:
        top_idx = torch.topk(patch_probs, k=min_keep).indices
        selected_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
        selected_mask[top_idx] = True

    selected_idx = torch.nonzero(selected_mask, as_tuple=False).view(-1).tolist()

    img_f_ga = forward_with_selected_patches(clip_model, device, patch_preln, selected_idx)
    probs_ga = (100 * img_f_ga @ text_features.T).softmax(-1)
    pred = torch.argmax(probs_ga, dim=-1).item()
    keep_pct = len(selected_idx) / max(1, n_tokens)

    return pred, len(selected_idx), keep_pct


@torch.no_grad()
def original_clip(model, proc, dataset, prompts, text_features, device: str):
    correct = 0
    start = time.time()
    for item in dataset:
        img = item["image"] if "image" in item else item["img"]
        label = item["label"] if "label" in item else item["fine_label"]
        img_input = proc(img).unsqueeze(0).to(device)
        img_f = model.encode_image(img_input)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        sim = (100 * img_f @ text_features.T).softmax(-1)
        pred = sim.argmax().item()
        correct += int(pred == label)
    elapsed = time.time() - start
    n = max(1, len(dataset))
    return correct / n, elapsed / n


def gpp_eval_adaptive(
    selector,
    outputs_are_probs: bool,
    clip_model,
    clip_proc,
    dataset,
    text_features,
    device: str,
    pred_thresh: float,
    min_keep_tokens: int,
    save_per_img_metrics: bool,
    save_dir: str,
):
    correct = 0
    keep_pcts = []
    start = time.time()

    records = []
    for item in dataset:
        img_start = time.time()
        img = item["image"] if "image" in item else item["img"]
        label = item["label"] if "label" in item else item["fine_label"]

        pred, keep_tokens, keep_pct = predict_with_selector(
            img=img,
            selector=selector,
            clip_model=clip_model,
            clip_proc=clip_proc,
            text_features=text_features,
            device=device,
            outputs_are_probs=outputs_are_probs,
            pred_thresh=pred_thresh,
            min_keep_tokens=min_keep_tokens,
        )
        keep_pcts.append(keep_pct)
        correct += int(pred == label)

        if save_per_img_metrics:
            records.append(
                {
                    "accuracy": int(pred == label),
                    "inf_time_per_img": time.time() - img_start,
                    "keep_tokens": keep_tokens,
                    "keep_pct": keep_pct * 100.0,
                }
            )

    n = max(1, len(dataset))
    elapsed = time.time() - start
    keep_tensor = torch.tensor(keep_pcts, dtype=torch.float32)
    keep_mean = float(keep_tensor.mean().item()) if keep_pcts else 0.0
    keep_std = float(keep_tensor.std(unbiased=False).item()) if keep_pcts else 0.0

    if save_per_img_metrics:
        os.makedirs(save_dir, exist_ok=True)
        per_img_metrics_path = os.path.join(save_dir, "per_img_metrics_adaptive.jsonl")
        for record in records:
            save_record_jsonl(record, per_img_metrics_path)

    return correct / n, elapsed / n, keep_mean, keep_std


def main():
    cfg_path = str(ROOT / "config" / "benchmark.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples = cfg["num_samples"]
    dataset_name = cfg["dataset_name"]
    model_id = cfg["model_id"]
    data_split = cfg["split"]
    gpp_ckpt = cfg["gpp_model_checkpoint"]
    mlp = cfg["mlp"]
    use_local_data = cfg.get("use_local_data", False)
    data_dir = cfg.get("data_dir")
    cache_dir = cfg.get("cache_dir")
    strategies = cfg.get("strategies", ["gpp"])

    pred_thresh = float(cfg.get("pred_thresh", 0.5))
    min_keep_tokens = int(cfg.get("min_keep_tokens", 1))
    save_per_img_metrics = bool(cfg.get("save_per_img_metrics", False))
    save_dir = cfg.get("save_dir", "./benchmark_results")

    if cfg.get("keep_pct", 0) not in (0, 0.0, None):
        print(
            "[adaptive_eval] keep_pct is ignored in adaptive mode. "
            "Patch count is model-decided via threshold/min_keep_tokens."
        )

    print(f"Evaluating on dataset: {dataset_name} | split: {data_split}")
    print(f"Model CKpt path: {gpp_ckpt}")
    print(f"Selection mode: threshold={pred_thresh:.3f}, min_keep_tokens={min_keep_tokens}")
    if num_samples:
        print(f"Evaluating on number of samples: {num_samples}")
    else:
        print(f"Evaluating on full dataset: {dataset_name}")

    if use_local_data:
        dataset, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        dataset, prompts = load_data_normal(dataset_name, num_samples, data_split)

    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float().eval()
    text_features = get_text_features(prompts, clip_model, device)

    records = []

    if "clip" in strategies:
        clip_acc, clip_time = original_clip(clip_model, clip_proc, dataset, prompts, text_features, device)
        records.append({"strategy": "clip", "keep_pct": 100.0, "accuracy": clip_acc, "avg_time": clip_time})

    if "gpp" in strategies:
        selector, outputs_are_probs = build_selector(mlp, device)
        state = torch.load(gpp_ckpt, map_location=device)
        selector.load_state_dict(state.get("model_state_dict", state))
        selector.eval()

        gpp_acc, gpp_time, keep_mean, keep_std = gpp_eval_adaptive(
            selector=selector,
            outputs_are_probs=outputs_are_probs,
            clip_model=clip_model,
            clip_proc=clip_proc,
            dataset=dataset,
            text_features=text_features,
            device=device,
            pred_thresh=pred_thresh,
            min_keep_tokens=min_keep_tokens,
            save_per_img_metrics=save_per_img_metrics,
            save_dir=save_dir,
        )
        print(f"Adaptive keep_pct mean/std: {keep_mean * 100:.3f}% / {keep_std * 100:.3f}%")

        records.append(
            {
                "strategy": "gpp",
                "keep_pct": keep_mean * 100.0,
                "accuracy": gpp_acc,
                "avg_time": gpp_time,
                "keep_pct_std": keep_std * 100.0,
            }
        )

    display_comparison_tables(records, strategies)


if __name__ == "__main__":
    main()
