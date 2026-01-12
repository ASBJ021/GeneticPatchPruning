import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import time

import yaml
import sys
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from gpp.dataset.data_utils import load_data_normal
from gpp.model.clip_model import load_clip, forward_with_selected_patches
from gpp.utils.visual_utils import plot_heatmap_overlay  # for consistent heatmap coloring


def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_ga_jsonl(path: str) -> Dict[int, Dict]:
    """Load GA results JSONL into a dict keyed by image_id.

    Each line is expected to contain at least:
      {"image_id": int, "selected_indices": List[int], ...}
    """
    mapping: Dict[int, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                img_id = int(rec.get("image_id"))
                mapping[img_id] = rec
            except Exception:
                continue
    return mapping


@torch.no_grad()
def get_text_features(model, device: str, prompts: List[str]) -> torch.Tensor:
    import clip
    toks = clip.tokenize(prompts).to(device)
    txt = model.encode_text(toks)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt


@torch.no_grad()
def get_image_feature_full(model, pixel_values: torch.Tensor) -> torch.Tensor:
    img_f = model.encode_image(pixel_values)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    return img_f


def topk_info(probs: torch.Tensor, labels: List[str], k: int = 5) -> List[Tuple[str, float]]:
    values, idxs = torch.topk(probs.squeeze(0), k)
    out = []
    for v, i in zip(values.tolist(), idxs.tolist()):
        out.append((labels[i], float(v)))
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image_copy(img: Image.Image, path: str) -> None:
    img.save(path)


def draw_boxes_on_image_with_preprocess(img: Image.Image, selected_idx: List[int], model, processor, save_path: str) -> None:
    """Draw patch boxes using CLIP's resize/crop parameters for correct grid mapping.

    This mirrors visual_utils.visualize_on_original but saves to a custom path and
    does not attempt to display a window.
    """
    from torchvision.transforms import Resize, CenterCrop

    # Extract resize/crop params from CLIP preprocess
    resize = next(t for t in processor.transforms if isinstance(t, Resize))
    crop = next(t for t in processor.transforms if isinstance(t, CenterCrop))
    target = resize.size if isinstance(resize.size, int) else resize.size[0]
    crop_s = crop.size if isinstance(crop.size, int) else crop.size[0]

    ow, oh = img.size
    # maintain aspect ratio then center-crop to crop_s
    if ow < oh:
        nw, nh = target, int(target * oh / ow)
    else:
        nw, nh = int(target * ow / oh), target

    left, top = (nw - crop_s) // 2, (nh - crop_s) // 2
    sx, sy = ow / nw, oh / nh

    # Grid is crop_s/16 for ViT-B/16
    grid = crop_s // 16
    draw = ImageDraw.Draw(img)
    for ii in selected_idx:
        r, c = divmod(int(ii), grid)
        x0, y0 = (c * 16 + left) * sx, (r * 16 + top) * sy
        draw.rectangle([x0, y0, x0 + 16 * sx, y0 + 16 * sy], outline="red", width=2)
    img.save(save_path)


def save_heatmap(img: Image.Image, selected_idx: List[int], grid_hw: Tuple[int, int], save_path: str, alpha: float = 0.5):
    """Create and save a heatmap overlay where selected patches are highlighted.

    Uses visual_utils.plot_heatmap_overlay for consistent look, while capturing
    the figure to disk.
    """
    import matplotlib.pyplot as plt

    H, W = grid_hw
    mask = np.zeros(H * W, dtype=np.float32)
    for i in selected_idx:
        if 0 <= int(i) < H * W:
            mask[int(i)] = 1.0

    # Prevent interactive window; capture figure and save
    prev_interactive = plt.isinteractive()
    try:
        plt.ioff()
        # plot_heatmap_overlay(img, mask, (H, W), alpha=alpha)
        fig = plt.gcf()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    finally:
        if prev_interactive:
            plt.ion()


def main():
    parser = argparse.ArgumentParser(description="Compare original CLIP vs GA-selected patches per image and save visuals.")
    parser.add_argument("--ga_jsonl", type=str, required=True, help="Path to GA results JSONL file")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), f"compare_outputs_{time.time()}"), help="Directory to save outputs")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of records to process (0=all)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    dataset_name = cfg.get("dataset_name", "clane9/imagenet-100")
    split = cfg.get("split", "validation")
    model_id = cfg.get("model_id", "ViT-B/16")
    num_samples = int(cfg.get("num_samples", 0))

    ensure_dir(args.out_dir)
    images_dir = os.path.join(args.out_dir, "images")
    ensure_dir(images_dir)

    # Load dataset and labels/prompts
    dataset, prompts = load_data_normal(dataset_name, num_samples, SPLIT=split)
    label_names = dataset.features["label"].names

    # Load CLIP
    model, processor = load_clip(model_id, device)
    model = model.float()
    text_features = get_text_features(model, device, prompts)

    # Load GA results
    ga_map = load_ga_jsonl(args.ga_jsonl)
    ids = sorted(ga_map.keys())
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    summary_rows = []
    probs_dir = os.path.join(args.out_dir, "probs")
    ensure_dir(probs_dir)

    og_total_acc = 0
    ga_total_acc = 0
    count = 0

    for idx in ids:
        rec = ga_map[idx]
        selected = rec.get("selected_indices", [])

        sample = dataset[int(idx)]
        img: Image.Image = sample["image"]
        gt = int(sample["label"]) if "label" in sample else int(rec.get("gt", -1))

        # Pixel tensor
        pixel_values = processor(img).unsqueeze(0).to(device)

        # Original CLIP probabilities
        img_f_full = get_image_feature_full(model, pixel_values)
        probs_og = (100 * img_f_full @ text_features.T).softmax(-1)

        # GA-selected probabilities
        # Compute pre-LN tokens from conv1 for forward_with_selected_patches
        with torch.no_grad():
            x = model.visual.conv1(pixel_values)
            B, D, H, W = x.shape
            x = x.view(B, D, -1).permute(0, 2, 1)

        img_f_ga = forward_with_selected_patches(model, device, x, selected)
        probs_ga = (100 * img_f_ga @ text_features.T).softmax(-1)

        # Top-1 info
        og_top1 = torch.argmax(probs_og, dim=-1).item()
        ga_top1 = torch.argmax(probs_ga, dim=-1).item()
        og_top1_prob = probs_og[0, og_top1].item()
        ga_top1_prob = probs_ga[0, ga_top1].item()

        if gt == og_top1:
            # print(f'Correct Prediction for sample {id}: {gt = } & {pred = }')
            og_total_acc+=1
            # print(f'og_total_acc = {og_total_acc}')
        if gt == ga_top1:
            # print(f'GA Correct Prediction for sample {id}: {gt = } & {pred = }')
            ga_total_acc+=1
        count+=1


        # Save original image
        # img_path = os.path.join(images_dir, f"{idx:05d}_orig.jpg")
        # save_image_copy(img, img_path)

        # # Save heatmap overlay (selected patches = 1.0)
        # heatmap_path = os.path.join(images_dir, f"{idx:05d}_heatmap.png")
        # save_heatmap(img.copy(), selected, (H, W), heatmap_path, alpha=0.5)

        # Save boxes overlay
        # boxes_path = os.path.join(images_dir, f"{idx:05d}_boxes.png")
        # draw_boxes_on_image_with_preprocess(img.copy(), selected, model, processor, boxes_path)

        # Save per-image top-5 probabilities JSON
        top5_og = topk_info(probs_og, label_names, k=5)
        top5_ga = topk_info(probs_ga, label_names, k=5)
        # with open(os.path.join(probs_dir, f"{idx:05d}.json"), "w", encoding="utf-8") as f:
        #     json.dump({
        #         "image_id": idx,
        #         "gt": gt,
        #         "og_top5": [(c, float(p)) for c, p in top5_og],
        #         "ga_top5": [(c, float(p)) for c, p in top5_ga],
        #         "og_top1": {"label": label_names[og_top1], "prob": float(og_top1_prob)},
        #         "ga_top1": {"label": label_names[ga_top1], "prob": float(ga_top1_prob)},
        #     }, f, indent=2)

        # summary_rows.append({
        #     "image_id": idx,
        #     "gt": gt,
        #     "og_top1": og_top1,
        #     "og_top1_label": label_names[og_top1],
        #     "og_top1_prob": og_top1_prob,
        #     "ga_top1": ga_top1,
        #     "ga_top1_label": label_names[ga_top1],
        #     "ga_top1_prob": ga_top1_prob,
        #     "selected_count": len(selected),
        # })

    # # Write summary CSV
    # import csv
    # csv_path = os.path.join(args.out_dir, "summary.csv")
    # if summary_rows:
    #     with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    #         writer.writeheader()
    #         writer.writerows(summary_rows)

    # print(f"Saved images to: {images_dir}")
    # print(f"Saved per-image probabilities to: {probs_dir}")
    # print(f"Saved summary CSV to: {csv_path}")

    print(f'Original CLIP Accuracy: {og_total_acc/count*100 :.2f} %')
    print(f'GA-based Patch Selector CLIP Accuracy: {ga_total_acc/count*100 :.2f} %')


if __name__ == "__main__":
    main()

