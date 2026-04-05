import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import clip
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpp.dataset.data_utils import load_data_folder, load_data_normal
from gpp.model.model import (
    LightweightPatchSelector,
    PatchSelector,
    PatchSelectorResBlock,
    PatchSelectorWithSoftmax,
    SimplePatchSelector,
    SimplePatchSelectorWithDropout,
)


def load_jsonl_annotations(path: str) -> Dict[int, List[int]]:
    id_to_selected: Dict[int, List[int]] = {}
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
            id_to_selected[image_id] = [int(x) for x in selected]
    return id_to_selected


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


def select_indices_from_probs(
    probs: torch.Tensor,
    pred_thresh: float,
    keep_pct_target: Optional[float],
    min_keep_tokens: int,
) -> List[int]:
    n_tokens = int(probs.numel())
    if n_tokens <= 0:
        return []

    # if keep_pct_target is not None and keep_pct_target > 0:
    #     keep_pct_target = float(max(0.0, min(1.0, keep_pct_target)))
    #     k = int(round(keep_pct_target * n_tokens))
    #     k = max(1, min(n_tokens, k))
    #     idx = torch.topk(probs, k=k, dim=-1).indices
    #     return torch.sort(idx).values.tolist()

    mask = probs >= pred_thresh
    if min_keep_tokens > 0 and int(mask.sum().item()) < min_keep_tokens:
        k = max(1, min(int(min_keep_tokens), n_tokens))
        idx = torch.topk(probs, k=k, dim=-1).indices
        return torch.sort(idx).values.tolist()

    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    return idx.tolist()


@torch.no_grad()
def predict_selected_indices(
    img,
    selector,
    clip_model,
    clip_proc,
    device: str,
    outputs_are_probs: bool,
    pred_thresh: float,
    keep_pct_target: Optional[float],
    min_keep_tokens: int,
) -> tuple[List[int], int]:
    pixel_values = clip_proc(img).unsqueeze(0).to(device)

    x = clip_model.visual.conv1(pixel_values)
    bsz, dim, _, _ = x.shape
    tokens = x.reshape(bsz, dim, -1).permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(bsz, -1, -1)
    tokens = torch.cat([cls_token, tokens], dim=1)
    tokens = tokens + clip_model.visual.positional_embedding[: tokens.size(1)].unsqueeze(0)

    tokens = clip_model.visual.ln_pre(tokens)
    tokens = clip_model.visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    patch_tokens = tokens[:, 1:, :]

    raw = selector(patch_tokens).squeeze(-1)
    probs = raw if outputs_are_probs else torch.sigmoid(raw)

    selected = select_indices_from_probs(
        probs=probs[0],
        pred_thresh=pred_thresh,
        keep_pct_target=keep_pct_target,
        min_keep_tokens=min_keep_tokens,
    )
    return selected, int(patch_tokens.shape[1])


def compute_overlap_metrics(pred: List[int], target: List[int], n_tokens: int) -> Dict[str, float]:
    pred_set = set(pred)
    tgt_set = set(target)

    inter = len(pred_set & tgt_set)
    union = len(pred_set | tgt_set)

    precision = inter / max(1, len(pred_set))
    recall = inter / max(1, len(tgt_set))
    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    iou = inter / max(1, union)
    exact_match = 1.0 if pred_set == tgt_set else 0.0

    return {
        "intersection": float(inter),
        "union": float(union),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "exact_match": float(exact_match),
        "pred_keep_tokens": float(len(pred_set)),
        "target_keep_tokens": float(len(tgt_set)),
        "pred_keep_pct": float(len(pred_set) * 100.0 / max(1, n_tokens)),
        "target_keep_pct": float(len(tgt_set) * 100.0 / max(1, n_tokens)),
        "abs_keep_token_diff": float(abs(len(pred_set) - len(tgt_set))),
    }


def mean_metric(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(r[key] for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Run patch selector and compare predicted selected patches against saved JSONL selected_indices."
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "benchmark.yaml"))
    parser.add_argument("--annotation_path", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    dataset_name = cfg["dataset_name"]
    data_split = cfg["split"]
    num_samples = int(cfg.get("num_samples", 0))
    use_local_data = bool(cfg.get("use_local_data", False))
    data_dir = cfg.get("data_dir")
    cache_dir = cfg.get("cache_dir")

    annotation_path = args.annotation_path.strip() or cfg.get("annotation_path", "")
    if not annotation_path:
        raise ValueError("annotation_path is required (arg --annotation_path or config annotation_path)")

    gpp_ckpt = cfg["gpp_model_checkpoint"]
    model_id = cfg["model_id"]
    mlp = cfg["mlp"]

    pred_thresh = float(cfg.get("selection_threshold", 0.5))
    # keep_pct_cfg = cfg.get("keep_pct", None)
    keep_pct_target = None
    # if keep_pct_cfg is not None:
    #     keep_pct_target = float(keep_pct_cfg)
    #     if keep_pct_target <= 0:
    #         keep_pct_target = None
    #     else:
    #         keep_pct_target = max(0.0, min(1.0, keep_pct_target))
    min_keep_tokens = int(cfg.get("min_keep_tokens", 1))

    save_dir = cfg.get("save_dir", "./benchmark_results")
    out_dir = os.path.join(save_dir, f"patch_compare_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    ann_map = load_jsonl_annotations(annotation_path)
    if not ann_map:
        raise ValueError(f"No usable records in annotation file: {annotation_path}")
    ann_ids_sorted = sorted(ann_map.keys())
    max_ann_id = ann_ids_sorted[-1]

    # Ensure we can access all annotation ids. If num_samples truncates the split,
    # automatically load the full split for correct id-based comparison.
    effective_num_samples = num_samples
    if num_samples > 0 and max_ann_id >= num_samples:
        print(
            "num_samples would truncate annotation ids. "
            f"max_annotation_id={max_ann_id}, num_samples={num_samples}. "
            "Loading full split instead."
        )
        effective_num_samples = 0

    print(
        f"Loading dataset={dataset_name}, split={data_split}, "
        f"num_samples={effective_num_samples} (requested={num_samples})"
    )
    if use_local_data:
        dataset, _ = load_data_folder(
            data_dir,
            effective_num_samples,
            SPLIT=data_split,
            cache_dir=cache_dir,
        )
    else:
        dataset, _ = load_data_normal(dataset_name, effective_num_samples, data_split)

    indices = [i for i in sorted(ann_map.keys()) if 0 <= i < len(dataset)]
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]

    if not indices:
        raise ValueError(
            "No overlapping image ids between annotation file and loaded dataset. "
            f"dataset_len={len(dataset)}, annotation_records={len(ann_map)}"
        )

    print(f"Using {len(indices)} comparable samples")

    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float().eval()

    selector, outputs_are_probs = build_selector(mlp, device)
    state = torch.load(gpp_ckpt, map_location=device)
    selector.load_state_dict(state.get("model_state_dict", state))
    selector.eval()

    print(f"Checkpoint: {gpp_ckpt}")
    if keep_pct_target is not None:
        print(f"Selection mode: top-k keep_pct={keep_pct_target:.4f}")
    else:
        print(f"Selection mode: threshold pred_thresh={pred_thresh:.4f}, min_keep_tokens={min_keep_tokens}")

    rows: List[Dict] = []
    for idx in tqdm(indices, desc="Comparing"):
        sample = dataset[int(idx)]
        img = sample["image"] if "image" in sample else sample["img"]
        target = ann_map[idx]

        pred, n_tokens = predict_selected_indices(
            img=img,
            selector=selector,
            clip_model=clip_model,
            clip_proc=clip_proc,
            device=device,
            outputs_are_probs=outputs_are_probs,
            pred_thresh=pred_thresh,
            keep_pct_target=keep_pct_target,
            min_keep_tokens=min_keep_tokens,
        )

        metrics = compute_overlap_metrics(pred, target, n_tokens)
        row = {
            "image_id": int(idx),
            "n_tokens": int(n_tokens),
            "pred_indices": pred,
            "target_indices": target,
            **metrics,
        }
        rows.append(row)

    summary = {
        "num_samples": len(rows),
        "annotation_path": annotation_path,
        "gpp_checkpoint": gpp_ckpt,
        "mlp": mlp,
        "selection_mode": "topk_keep_pct" if keep_pct_target is not None else "threshold",
        "keep_pct_target": keep_pct_target,
        "pred_thresh": pred_thresh,
        "min_keep_tokens": min_keep_tokens,
        "mean_intersection": mean_metric(rows, "intersection"),
        "mean_union": mean_metric(rows, "union"),
        "mean_iou": mean_metric(rows, "iou"),
        "mean_precision": mean_metric(rows, "precision"),
        "mean_recall": mean_metric(rows, "recall"),
        "mean_f1": mean_metric(rows, "f1"),
        "exact_match_rate": mean_metric(rows, "exact_match"),
        "mean_pred_keep_pct": mean_metric(rows, "pred_keep_pct"),
        "mean_target_keep_pct": mean_metric(rows, "target_keep_pct"),
        "mean_abs_keep_token_diff": mean_metric(rows, "abs_keep_token_diff"),
    }

    per_image_path = os.path.join(out_dir, "per_image_patch_compare.jsonl")
    with open(per_image_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPatch Selection Comparison Summary")
    print(f"samples: {summary['num_samples']}")
    print(f'mean intersection: {summary["mean_intersection"]:.2f}')
    print(f"mean union: {summary['mean_union']:.2f}")
    print(f"mean IoU: {summary['mean_iou']:.4f}")
    print(f"mean precision: {summary['mean_precision']:.4f}")
    print(f"mean recall: {summary['mean_recall']:.4f}")
    print(f"mean F1: {summary['mean_f1']:.4f}")
    print(f"exact-match rate: {summary['exact_match_rate']:.4f}")
    print(f"mean pred keep%: {summary['mean_pred_keep_pct']:.2f}")
    print(f"mean target keep%: {summary['mean_target_keep_pct']:.2f}")
    print(f"mean |keep token diff|: {summary['mean_abs_keep_token_diff']:.2f}")
    print(f"Saved per-image metrics to: {per_image_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
