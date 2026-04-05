from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import clip
import numpy as np
import torch
import yaml
from tqdm import tqdm

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


def otsu_intraclass_variance(values: np.ndarray, threshold: float) -> float:
    hi = values >= threshold
    lo = values < threshold

    if not hi.any() and not lo.any():
        return float("inf")

    w_hi = float(np.mean(hi)) if hi.size > 0 else 0.0
    w_lo = float(np.mean(lo)) if lo.size > 0 else 0.0
    v_hi = float(np.var(values, where=hi)) if hi.any() else 0.0
    v_lo = float(np.var(values, where=lo)) if lo.any() else 0.0
    return w_hi * v_hi + w_lo * v_lo


def otsu_threshold_from_probs(probs_1d: torch.Tensor) -> float:
    values = probs_1d.detach().float().clamp(0.0, 1.0).cpu().numpy().reshape(-1)
    if values.size == 0:
        return 0.5
    uniq = np.unique(values)
    if uniq.size <= 1:
        return float(uniq[0]) if uniq.size == 1 else 0.5

    candidates = (uniq[:-1] + uniq[1:]) / 2.0
    best_threshold = float(candidates[0])
    best_score = float("inf")
    for th in candidates:
        score = otsu_intraclass_variance(values, float(th))
        if score < best_score:
            best_score = score
            best_threshold = float(th)
    return best_threshold


def otsu_select_indices(
    probs_1d: torch.Tensor,
    min_keep_tokens: int,
    max_keep_tokens: int,
) -> List[int]:
    n_tokens = int(probs_1d.numel())
    if n_tokens <= 0:
        return []

    threshold = otsu_threshold_from_probs(probs_1d)
    idx = torch.nonzero(probs_1d >= threshold, as_tuple=False).squeeze(-1)
    selected = torch.sort(idx).values.tolist() if idx.numel() > 0 else []

    min_keep = max(1, min(int(min_keep_tokens), n_tokens))
    if max_keep_tokens <= 0:
        max_keep = n_tokens
    else:
        max_keep = max(min_keep, min(int(max_keep_tokens), n_tokens))

    if len(selected) < min_keep:
        selected = torch.topk(probs_1d, k=min_keep, dim=-1).indices
        selected = torch.sort(selected).values.tolist()
    elif len(selected) > max_keep:
        selected = torch.topk(probs_1d, k=max_keep, dim=-1).indices
        selected = torch.sort(selected).values.tolist()

    return selected


@torch.no_grad()
def predict_with_selector_otsu(
    img,
    selector,
    clip_model,
    clip_proc,
    text_features: torch.Tensor,
    device: str,
    outputs_are_probs: bool,
    min_keep_tokens: int,
    max_keep_tokens: int,
):
    pixel_values = clip_proc(img).unsqueeze(0).to(device)

    x = clip_model.visual.conv1(pixel_values)
    bsz, dim, _, _ = x.shape
    patch_preln = x.reshape(bsz, dim, -1).permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(bsz, -1, -1)
    tokens = torch.cat([cls_token, patch_preln], dim=1)
    tokens = tokens + clip_model.visual.positional_embedding[: tokens.size(1)].unsqueeze(0)
    tokens = clip_model.visual.ln_pre(tokens)
    tokens = clip_model.visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)

    patch_tokens = tokens[:, 1:, :]
    raw = selector(patch_tokens).squeeze(-1)
    probs = raw if outputs_are_probs else torch.sigmoid(raw)
    selected_idx = otsu_select_indices(
        probs[0],
        min_keep_tokens=min_keep_tokens,
        max_keep_tokens=max_keep_tokens,
    )

    img_f = forward_with_selected_patches(clip_model, device, patch_preln, selected_idx)
    probs_cls = (100 * img_f @ text_features.T).softmax(-1)
    pred = torch.argmax(probs_cls, dim=-1).item()

    n_tokens = int(patch_tokens.size(1))
    keep_pct = len(selected_idx) / max(1, n_tokens)
    return pred, selected_idx, keep_pct


@torch.no_grad()
def original_clip(model, proc, dataset, text_features, device: str):
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


def load_annotation_map(path: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not path or not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "image_id" not in rec:
                continue
            image_id = int(rec["image_id"])
            selected = rec.get("selected_indices", []) or []
            out[image_id] = [int(x) for x in selected]
    return out


def overlap_metrics(pred_idx: List[int], tgt_idx: List[int]) -> Dict[str, float]:
    p = set(pred_idx)
    t = set(tgt_idx)
    inter = len(p & t)
    union = len(p | t)

    precision = inter / max(1, len(p))
    recall = inter / max(1, len(t))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = inter / max(1, union)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": float(1.0 if p == t else 0.0),
        "abs_keep_diff": float(abs(len(p) - len(t))),
    }


def mean_metric(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(r[key] for r in rows) / len(rows))


def gpp_eval_otsu(
    selector,
    outputs_are_probs: bool,
    clip_model,
    clip_proc,
    dataset,
    text_features,
    device: str,
    min_keep_tokens: int,
    max_keep_tokens: int,
    save_per_img_metrics: bool,
    save_dir: str,
    annotation_map: Optional[Dict[int, List[int]]] = None,
):
    correct = 0
    keep_pcts = []
    start = time.time()

    records = []
    overlap_rows: List[Dict] = []

    for i, item in enumerate(tqdm(dataset, desc="Otsu eval", leave=False)):
        img_start = time.time()
        img = item["image"] if "image" in item else item["img"]
        label = item["label"] if "label" in item else item["fine_label"]

        pred, selected_idx, keep_pct = predict_with_selector_otsu(
            img=img,
            selector=selector,
            clip_model=clip_model,
            clip_proc=clip_proc,
            text_features=text_features,
            device=device,
            outputs_are_probs=outputs_are_probs,
            min_keep_tokens=min_keep_tokens,
            max_keep_tokens=max_keep_tokens,
        )

        is_correct = int(pred == label)
        keep_pcts.append(keep_pct)
        correct += is_correct

        row = {
            "image_id": i,
            "accuracy": is_correct,
            "inf_time_per_img": time.time() - img_start,
            "keep_tokens": int(len(selected_idx)),
            "keep_pct": float(keep_pct * 100.0),
            "selected_indices": selected_idx,
        }

        if annotation_map is not None and i in annotation_map:
            om = overlap_metrics(selected_idx, annotation_map[i])
            row.update({f"overlap_{k}": v for k, v in om.items()})
            overlap_rows.append(om)

        records.append(row)

    n = max(1, len(dataset))
    elapsed = time.time() - start
    keep_tensor = torch.tensor(keep_pcts, dtype=torch.float32)
    keep_mean = float(keep_tensor.mean().item()) if keep_pcts else 0.0
    keep_std = float(keep_tensor.std(unbiased=False).item()) if keep_pcts else 0.0

    if save_per_img_metrics:
        os.makedirs(save_dir, exist_ok=True)
        per_img_path = os.path.join(save_dir, "per_img_metrics_otsu.jsonl")
        for r in records:
            save_record_jsonl(r, per_img_path)

    overlap_summary = None
    if overlap_rows:
        overlap_summary = {
            "mean_iou": mean_metric(overlap_rows, "iou"),
            "mean_precision": mean_metric(overlap_rows, "precision"),
            "mean_recall": mean_metric(overlap_rows, "recall"),
            "mean_f1": mean_metric(overlap_rows, "f1"),
            "exact_match_rate": mean_metric(overlap_rows, "exact_match"),
            "mean_abs_keep_diff": mean_metric(overlap_rows, "abs_keep_diff"),
        }
        summary_path = os.path.join(save_dir, "otsu_overlap_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(overlap_summary, f, indent=2)

    return correct / n, elapsed / n, keep_mean, keep_std, overlap_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate selector with Otsu patch selection.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "benchmark_ga_distill.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples = int(cfg["num_samples"])
    dataset_name = cfg["dataset_name"]
    model_id = cfg["model_id"]
    data_split = cfg["split"]
    gpp_ckpt = cfg["gpp_model_checkpoint"]
    mlp = cfg["mlp"]
    use_local_data = bool(cfg.get("use_local_data", False))
    data_dir = cfg.get("data_dir")
    cache_dir = cfg.get("cache_dir")
    strategies = cfg.get("strategies", ["gpp"])
    save_per_img_metrics = bool(cfg.get("save_per_img_metrics", False))
    save_dir = cfg.get("save_dir", "./benchmark_results")

    selection_cfg = cfg.get("selection", {})
    min_keep_tokens = int(selection_cfg.get("min_keep_tokens", 1))
    max_keep_tokens = int(selection_cfg.get("max_keep_tokens", 0))

    annotation_path = cfg.get("annotation_path", "")
    annotation_map = load_annotation_map(annotation_path) if annotation_path else {}
    if annotation_map:
        print(f"Loaded {len(annotation_map)} annotation ids for overlap metrics")

    print(f"Evaluating on dataset: {dataset_name} | split: {data_split}")
    print(f"Checkpoint: {gpp_ckpt}")
    print(
        f"Selection mode: otsu_intraclass_variance (min_keep={min_keep_tokens}, max_keep={max_keep_tokens})"
    )

    if use_local_data:
        dataset, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        dataset, prompts = load_data_normal(dataset_name, num_samples, data_split)

    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float().eval()
    text_features = get_text_features(prompts, clip_model, device)

    records = []

    if "clip" in strategies:
        clip_acc, clip_time = original_clip(clip_model, clip_proc, dataset, text_features, device)
        records.append({"strategy": "clip", "keep_pct": 100.0, "accuracy": clip_acc, "avg_time": clip_time})

    if "gpp" in strategies:
        selector, outputs_are_probs = build_selector(mlp, device)
        state = torch.load(gpp_ckpt, map_location=device)
        selector.load_state_dict(state.get("model_state_dict", state))
        selector.eval()

        gpp_acc, gpp_time, keep_mean, keep_std, overlap_summary = gpp_eval_otsu(
            selector=selector,
            outputs_are_probs=outputs_are_probs,
            clip_model=clip_model,
            clip_proc=clip_proc,
            dataset=dataset,
            text_features=text_features,
            device=device,
            min_keep_tokens=min_keep_tokens,
            max_keep_tokens=max_keep_tokens,
            save_per_img_metrics=save_per_img_metrics,
            save_dir=save_dir,
            annotation_map=annotation_map if annotation_map else None,
        )

        print(f"Otsu keep_pct mean/std: {keep_mean * 100:.3f}% / {keep_std * 100:.3f}%")
        if overlap_summary is not None:
            print(
                "Overlap summary: "
                f"IoU={overlap_summary['mean_iou']:.4f}, "
                f"F1={overlap_summary['mean_f1']:.4f}, "
                f"Exact={overlap_summary['exact_match_rate']:.4f}"
            )

        records.append(
            {
                "strategy": "gpp_otsu",
                "keep_pct": keep_mean * 100.0,
                "accuracy": gpp_acc,
                "avg_time": gpp_time,
                "keep_pct_std": keep_std * 100.0,
            }
        )

    display_comparison_tables(records, strategies)


if __name__ == "__main__":
    main()
