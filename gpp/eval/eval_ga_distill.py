from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Resize

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_cuda_device(device: str) -> bool:
    return torch.cuda.is_available() and str(device).startswith("cuda")


def cuda_peak_delta_mb(device: str, baseline_allocated: int = 0) -> float:
    """Return peak allocated delta in MB relative to baseline allocation."""
    peak = torch.cuda.max_memory_allocated(device)
    return max(0.0, float(peak - baseline_allocated) / (1024 ** 2))


def build_selector(mlp: str, device: str, mixer_mlp_ratio):
    outputs_are_probs = False
    if mlp == "PatchSelector":
        selector = PatchSelector(mixer_mlp_ratio=mixer_mlp_ratio).to(device)
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


def adaptive_topk_select_indices(
    probs_1d: torch.Tensor,
    min_keep_tokens: int,
    max_keep_tokens: int,
) -> List[int]:
    n_tokens = int(probs_1d.numel())
    if n_tokens <= 0:
        return []

    min_keep = max(1, min(int(min_keep_tokens), n_tokens))
    if max_keep_tokens <= 0:
        max_keep = n_tokens
    else:
        max_keep = max(min_keep, min(int(max_keep_tokens), n_tokens))

    soft_k = int(torch.round(probs_1d.sum()).item())
    k = max(min_keep, min(max_keep, soft_k))

    idx = torch.topk(probs_1d, k=k, dim=-1).indices
    return torch.sort(idx).values.tolist()


def save_original_clip_heatmap(
    img: Image.Image,
    patch_preln: torch.Tensor,
    clip_model,
    pred_text_feature: torch.Tensor,
    save_path: str,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> None:
    patch_tokens = clip_model.visual.ln_pre(
        patch_preln + clip_model.visual.positional_embedding[1 : patch_preln.size(1) + 1].unsqueeze(0)
    )
    patch_embed = patch_tokens @ clip_model.visual.proj if clip_model.visual.proj is not None else patch_tokens
    patch_embed = patch_embed / patch_embed.norm(dim=-1, keepdim=True)
    sims = torch.matmul(patch_embed[0], pred_text_feature.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()

    num_patches = int(sims.shape[0])
    grid = int(round(num_patches ** 0.5))
    heatmap = sims.reshape(grid, grid)
    rng = float(np.ptp(heatmap))
    norm = (heatmap - heatmap.min()) / (rng + 1e-8)

    if img.mode != "RGB":
        img = img.convert("RGB")
    base = np.array(img, dtype=np.float32)
    h, w = base.shape[:2]
    hm = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
    colored = plt.get_cmap(cmap)(np.array(hm) / 255.0)[..., :3]
    blended = (base * (1 - alpha) + colored * 255 * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(save_path)


def draw_boxes_on_image_with_preprocess(
    img: Image.Image,
    selected_idx: List[int],
    processor,
    save_path: str,
    outline_color: str = "red",
) -> None:
    resize = next(t for t in processor.transforms if isinstance(t, Resize))
    crop = next(t for t in processor.transforms if isinstance(t, CenterCrop))
    target = resize.size if isinstance(resize.size, int) else resize.size[0]
    crop_s = crop.size if isinstance(crop.size, int) else crop.size[0]

    ow, oh = img.size
    if ow < oh:
        nw, nh = target, int(target * oh / ow)
    else:
        nw, nh = int(target * ow / oh), target

    left, top = (nw - crop_s) // 2, (nh - crop_s) // 2
    sx, sy = ow / nw, oh / nh

    grid = crop_s // 16
    draw = ImageDraw.Draw(img)
    for ii in selected_idx:
        r, c = divmod(int(ii), grid)
        x0, y0 = (c * 16 + left) * sx, (r * 16 + top) * sy
        draw.rectangle([x0, y0, x0 + 16 * sx, y0 + 16 * sy], outline=outline_color, width=2)
    img.save(save_path)


@torch.no_grad()
def predict_with_selector_adaptive_topk(
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
    baseline_allocated = 0
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
    sel_start = time.time()
    raw = selector(patch_tokens).squeeze(-1)
    probs = raw if outputs_are_probs else torch.sigmoid(raw)
    selected_idx = adaptive_topk_select_indices(probs[0], min_keep_tokens, max_keep_tokens)
    sel_end = time.time() - sel_start

    if is_cuda_device(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = 0

    img_f = forward_with_selected_patches(clip_model, device, patch_preln, selected_idx) # image features after patch selection

    if is_cuda_device(device):
        torch.cuda.synchronize()
        post_select_vram_mb = cuda_peak_delta_mb(device, baseline_allocated)
    else:
        post_select_vram_mb = 0.0
    probs_cls = (100 * img_f @ text_features.T).softmax(-1)
    pred = torch.argmax(probs_cls, dim=-1).item()

    n_tokens = int(patch_tokens.size(1))
    keep_pct = len(selected_idx) / max(1, n_tokens)
    return pred, selected_idx, keep_pct, sel_end, post_select_vram_mb, patch_preln


@torch.no_grad()
def original_clip(model, proc, dataset, text_features, device: str):
    correct = 0
    start = time.time()
    mem = 0
    for item in dataset:
        img = item["image"] if "image" in item else item["img"]
        label = item["label"] if "label" in item else item["fine_label"]

       

        img_input = proc(img).unsqueeze(0).to(device)

        if is_cuda_device(device):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            baseline_allocated = torch.cuda.memory_allocated(device)
        else:
            baseline_allocated = 0

        img_f = model.encode_image(img_input)

        if is_cuda_device(device):
            torch.cuda.synchronize()
            peak_vram_mb = cuda_peak_delta_mb(device, baseline_allocated)
            mem = mem + peak_vram_mb

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        sim = (100 * img_f @ text_features.T).softmax(-1)
        

        pred = sim.argmax().item()
        correct += int(pred == label)

    elapsed = time.time() - start
    n = max(1, len(dataset))
    return correct / n, elapsed / n, mem / n


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


def gpp_eval_adaptive_topk(
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
    visualize: bool,
    annotation_map: Optional[Dict[int, List[int]]] = None,
):
    correct = 0
    keep_pcts = []
    start = time.time()

    records = []
    overlap_rows: List[Dict] = []
    mem = 0
    vis_dir = os.path.join(save_dir, "images")
    if visualize:
        ensure_dir(vis_dir)

    for i, item in enumerate(tqdm(dataset, desc="Adaptive eval", leave=False)):

        img = item["image"] if "image" in item else item["img"]
        label = item["label"] if "label" in item else item["fine_label"]

        # if device == "cuda":
        #     torch.cuda.reset_peak_memory_stats(device)
        #     torch.cuda.synchronize()
        img_start = time.time()

        pred, selected_idx, keep_pct, sel_time, post_select_vram_mb, patch_preln = predict_with_selector_adaptive_topk(
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
        # if device == "cuda":
        #     torch.cuda.synchronize() 
        #     peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        #     mem = mem + peak_vram_mb
        mem = mem + post_select_vram_mb

        is_correct = int(pred == label)
        keep_pcts.append(keep_pct)
        correct += is_correct

        end_time = time.time() - img_start

        row = {
            "image_id": i,
            "pred": int(pred),
            "gt": int(label),
            "accuracy": is_correct,
            "inf_time_per_img": end_time,
            "mem_usage_image_encoder": post_select_vram_mb,
            "keep_tokens": int(len(selected_idx)),
            "keep_pct": float(keep_pct * 100.0),
            "selected_indices": selected_idx,
            "selector_time": sel_time,
        }

        if annotation_map is not None and i in annotation_map:
            om = overlap_metrics(selected_idx, annotation_map[i])
            row.update({f"overlap_{k}": v for k, v in om.items()})
            overlap_rows.append(om)

        if visualize:
            orig_img_path = os.path.join(vis_dir, f"{i:05d}_orig.png")
            heatmap_path = os.path.join(vis_dir, f"{i:05d}_orig_clip_heatmap.png")
            boxes_path = os.path.join(vis_dir, f"{i:05d}_selected_boxes.png")
            img.copy().save(orig_img_path)
            pred_text_feature = text_features[int(pred)].detach()
            save_original_clip_heatmap(
                img=img.copy(),
                patch_preln=patch_preln,
                clip_model=clip_model,
                pred_text_feature=pred_text_feature,
                save_path=heatmap_path,
            )
            draw_boxes_on_image_with_preprocess(
                img=img.copy(),
                selected_idx=selected_idx,
                processor=clip_proc,
                save_path=boxes_path,
                outline_color="green" if is_correct else "red",
            )
            row["orig_img_path"] = orig_img_path
            row["orig_clip_heatmap_path"] = heatmap_path
            row["selected_boxes_path"] = boxes_path

        records.append(row)

    n = max(1, len(dataset))
    elapsed = time.time() - start
    keep_tensor = torch.tensor(keep_pcts, dtype=torch.float32)
    keep_mean = float(keep_tensor.mean().item()) if keep_pcts else 0.0
    keep_std = float(keep_tensor.std(unbiased=False).item()) if keep_pcts else 0.0

    if save_per_img_metrics:
        os.makedirs(save_dir, exist_ok=True)
        per_img_path = os.path.join(save_dir, f"per_img_metrics_adaptive_topk_{len(records)}.jsonl")
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
        summary_path = os.path.join(save_dir, "adaptive_topk_overlap_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(overlap_summary, f, indent=2)

    return correct / n, elapsed / n, keep_mean, keep_std, overlap_summary, mem / n


def main():
    parser = argparse.ArgumentParser(description="Evaluate selector with GA-aligned adaptive top-k patch selection.")
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
    mixer_mlp_ratio = cfg.get("mixer_mlp_ratio", None)
    use_local_data = bool(cfg.get("use_local_data", False))
    data_dir = cfg.get("data_dir")
    cache_dir = cfg.get("cache_dir")
    strategies = cfg.get("strategies", ["gpp"])
    save_per_img_metrics = bool(cfg.get("save_per_img_metrics", False))
    save_dir = cfg.get("save_dir", "./benchmark_results")
    visualize = bool(cfg.get("visualize", False))

    selection_cfg = cfg.get("selection", {})
    min_keep_tokens = int(selection_cfg.get("min_keep_tokens", cfg.get("min_keep_tokens", 1)))
    max_keep_tokens = int(selection_cfg.get("max_keep_tokens", cfg.get("max_keep_tokens", 0)))

    annotation_path = cfg.get("annotation_path", "")
    annotation_map = load_annotation_map(annotation_path) if annotation_path else {}
    if annotation_map:
        print(f"Loaded {len(annotation_map)} annotation ids for overlap metrics")

    print(f"Evaluating on dataset: {dataset_name} | split: {data_split}")
    print(f"Checkpoint: {gpp_ckpt}")
    print(f"Selection mode: adaptive_topk (min_keep={min_keep_tokens}, max_keep={max_keep_tokens})")

    if use_local_data:
        dataset, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        dataset, prompts = load_data_normal(dataset_name, num_samples, data_split)

    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float().eval()
    text_features = get_text_features(prompts, clip_model, device)

    records = []

    if "clip" in strategies:
        print("Evaluating original CLIP...")
        clip_acc, clip_time, clip_mem = original_clip(clip_model, clip_proc, dataset, text_features, device)
        print(f'CLIP Average Memory Usage Image Encoder: {clip_mem:.2f} MB')
        records.append({"strategy": "clip", "keep_pct": 100.0, "accuracy": clip_acc, "avg_time": clip_time, "avg_memory_usage": clip_mem})

    if "gpp" in strategies:
        print("Evaluating GPP adaptive top-k selector...")
        selector, outputs_are_probs = build_selector(mlp, device, mixer_mlp_ratio)
        state = torch.load(gpp_ckpt, map_location=device)
        selector.load_state_dict(state.get("model_state_dict", state))
        selector.eval()

        gpp_acc, gpp_time, keep_mean, keep_std, overlap_summary, gpp_mem = gpp_eval_adaptive_topk(
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
            visualize=visualize,
            annotation_map=annotation_map if annotation_map else None,
        )

        print(f"Adaptive keep_pct mean/std: {keep_mean * 100:.3f}% / {keep_std * 100:.3f}%")
        print(f'GPP Average Memory Usage: {gpp_mem:.2f} MB')
        if overlap_summary is not None:
            print(
                "Overlap summary: "
                f"IoU={overlap_summary['mean_iou']:.4f}, "
                f"F1={overlap_summary['mean_f1']:.4f}, "
                f"Exact={overlap_summary['exact_match_rate']:.4f}"
            )

        records.append(
            {
                "strategy": "gpp_adaptive_topk",
                "keep_pct": keep_mean * 100.0,
                "accuracy": gpp_acc,
                "avg_time": gpp_time,
                "keep_pct_std": keep_std * 100.0,
                "avg_memory_usage": gpp_mem,
            }
        )

    display_comparison_tables(records, strategies)


if __name__ == "__main__":
    main()
