from typing import List, Tuple, Optional
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
import time
import numpy as np
# Ensure project root is available on sys.path when running the script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import yaml
from timm.models.mlp_mixer import MixerBlock
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from gpp.dataset.data_utils import load_data_normal, build_dataloaders, PatchIndexDataset, _load_jsonl, split_dataset
from gpp.model.model import PatchSelector, images_to_patches
from gpp.model.clip_model import forward_with_selected_patches, load_clip
from gpp.utils.visual_utils import plot_heatmap_overlay 

cfg_path = '/var/lit2425/jenga/GeneticPatchPruning/config/training_config.yaml'
print (f'Loading config from {cfg_path}')

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("device", "cuda")
if not torch.cuda.is_available():
    device = "cpu"


to_pil = transforms.ToPILImage()
exp_name = cfg["exp_name"]
num_samples  = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_split = cfg["split"]
model_id     = cfg["model_id"]
vis = cfg["visualize"]
# annotation_path = cfg["annotation_path"]
# img_size = cfg["img_size"]
batch_size = cfg["batch_size"]
seed = cfg["seed"]
# num_workers = cfg["num_workers"]
# epochs = cfg["epochs"]
# save_dir = cfg["save_dir"]
# exp_name = cfg["exp_name"]

ckpt_path = cfg["model_checkpoint"]
print(f'{ckpt_path = }')

clip_model, processor = clip.load(model_id, device)  # Load on CPU initially
clip_model.float()  # Ensure model is in float32

@torch.no_grad()
def get_text_features(prompts, model=clip_model) -> torch.Tensor:
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


def img_to_patch(img, selector, text_features, thresh=0.5):
    
    # pil_img = to_pil(img)
    pixel_values = processor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        x = clip_model.visual.conv1(pixel_values)
        B, D, H, W = x.shape
        x = x.reshape(B, D, -1).permute(0, 2, 1)

        cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, x], dim=1)
        tokens += clip_model.visual.positional_embedding.unsqueeze(0)

        tokens = clip_model.visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = clip_model.visual.transformer(tokens)
        tokens = tokens.permute(1, 0, 2)

        orig_cls_token = tokens[:, 0, :]
        orig_cls_token = clip_model.visual.ln_post(orig_cls_token)

        if clip_model.visual.proj is not None:
            orig_cls_token = orig_cls_token @ clip_model.visual.proj

        orig_cls_token /= orig_cls_token.norm(dim=-1, keepdim=True)
        patch_tokens = tokens[:, 1:, :]

        logits = selector(patch_tokens).squeeze(-1) 
        # print(f'{logits.shape = }')
        preds = (torch.sigmoid(logits) >= thresh).float()
        # print(f'{preds = }')

        selectd_idx  = torch.nonzero(preds[0], as_tuple=False).view(-1).tolist()
        # print(f'{selectd_idx = }')

        img_f_ga = forward_with_selected_patches(clip_model, device, x, selectd_idx)
        probs_ga = (100 * img_f_ga @ text_features.T).softmax(-1)

        # og_top1 = torch.argmax(probs_og, dim=-1).item()
        ga_top1 = torch.argmax(probs_ga, dim=-1).item()
        # og_top1_prob = probs_og[0, og_top1].item()
        ga_top1_prob = probs_ga[0, ga_top1].item()

    pred = ga_top1
    return ga_top1



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
        plot_heatmap_overlay(img, mask, (H, W), alpha=alpha)
        fig = plt.gcf()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    finally:
        if prev_interactive:
            plt.ion()

def main():
    
    parser = argparse.ArgumentParser(description="Compare original CLIP vs GA-selected patches per image and save visuals.")
    # parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), f"results/compare_outputs_{exp_name}_{time.time()}"), help="Directory to save outputs")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of records to process (0=all)")
    args = parser.parse_args()


    # patchselector = PatchSelector()
    # ckpt = torch.load(ckpt_path)
    print(f'Evaluating on number of samples: {num_samples}')
    selector = PatchSelector().to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    selector.load_state_dict(state.get("model_state_dict", state))
    selector.eval()
    # print(f'{selector = }')

    ds, prompts = load_data_normal("clane9/imagenet-100", NUM_SAMPLES=num_samples, SPLIT=data_split)
    # sample = ds[120]
    text_features = get_text_features(prompts)
    print(f'{text_features.shape = }')

    # img = sample['image']
    # gt = int(sample["label"])
    # pil_img = to_pil(img)

    # pred = img_to_patch(img, selector, text_features)
    # patches = torch.cat(patches, dim=0).to(device)

    # print(f'{gt = } & {pred = }')
    og_total_acc = 0
    ga_total_acc = 0
    count = 0
    start = 0


    for id in range(start, num_samples):
        sample = ds[int(id)]
        img = sample['image']
        gt = int(sample["label"])
        pixel_values = processor(img).unsqueeze(0).to(device)


        # Original CLIP probabilities
        img_f_full = get_image_feature_full(clip_model, pixel_values)
        probs_og = (100 * img_f_full @ text_features.T).softmax(-1)



        pred_ga = img_to_patch(img, selector, text_features)

        pred_og = torch.argmax(probs_og, dim=-1).item()
        # print(f'Image {id}: GT: {gt} | OG Pred: {og_top1} | GA Pred: {pred_ga}')
        # ga_top1 = torch.argmax(probs_ga, dim=-1).item()
        # pred_og = probs_og[0, og_top1].item()
        # print(f'{pred_og = }')
        
        if gt == pred_og:
            # print(f'Correct Prediction for sample {id}: {gt = } & {pred = }')
            og_total_acc+=1
            # print(f'og_total_acc = {og_total_acc}')
        if gt == pred_ga:
            # print(f'GA Correct Prediction for sample {id}: {gt = } & {pred = }')
            ga_total_acc+=1
        count+=1

    
    print(f'Original CLIP Accuracy: {og_total_acc/count*100 :.2f} %')
    print(f'GA-based Patch Selector CLIP Accuracy: {ga_total_acc/count*100 :.2f} %')


    
if __name__ == "__main__":
    main()