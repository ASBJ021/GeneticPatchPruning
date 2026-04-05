import os
import yaml
import torch
import pandas as pd
from pathlib import Path
import sys
# Ensure project root is available on sys.path when running the script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpp.dataset.data_utils import load_data_normal, load_data_folder
# from gpp.model.clip_model import original_clip
from gpp.eval.compare import display_comparison_tables
from gpp.model.model import PatchSelector, images_to_patches, SimplePatchSelector, PatchSelectorWithSoftmax, SimplePatchSelectorWithDropout, PatchSelectorResBlock, LightweightPatchSelector
from gpp.model.clip_model import forward_with_selected_patches, load_clip
from gpp.eval.evaluate import img_to_patch, get_image_feature_full, get_text_features, img_to_patch_with_sigmoid
from gpp.utils.visual_utils import plot_heatmap_overlay  # for consistent heatmap coloring
from gpp.eval.compare_og_ga import save_heatmap, save_image_copy, draw_boxes_on_image_with_preprocess
from gpp.genetic_algo.io import save_record_jsonl
from gpp.genetic_algo.image_gradent_selection import image_gradient_patch_selection

import clip
import time

def original_clip(model, proc, dataset, prompts, txt_f ,MODEL_ID, DEVICE):
    # model, proc = clip.load(MODEL_ID, DEVICE)
    # model = model.float()
    total, correct = 0.0, 0
    start = time.time()
    for item in dataset:

        img = item["image"] if "image" in item.keys() else item["img"]
        label = item["label"] if "label" in item.keys() else item["fine_label"]

        
        # img, label = item["image"], item["label"]


        img_input = proc(img).unsqueeze(0).to(DEVICE)
        toks = clip.tokenize(prompts).to(DEVICE)
        
        with torch.no_grad():
            img_f = model.encode_image(img_input)
            img_f /= img_f.norm(dim=-1,keepdim=True)
            txt_f /= txt_f.norm(dim=-1,keepdim=True)
            sim = (100*img_f@txt_f.T).softmax(-1)
            pred = sim.argmax().item()
        correct += (pred==label)
    end_time = time.time()
    total += end_time - start
    
    return correct/len(dataset), total/len(dataset)

def gpp_eval(selector, clip_model, clip_proc, dataset, prompts, text_features, MODEL_ID, DEVICE, save_per_img_metrics, save_dir, vis, selection_threshold):
    total_time, correct = 0.0, 0
    total_keep_pct = 0.0
    start_time = time.time()

    records = []
    img_dir = os.path.join(save_dir, "images")
    if vis:
        os.makedirs(img_dir, exist_ok=True)

    idx = 0
    print(f'selection_threshold = {selection_threshold}')

    for item in dataset:
        # img, label = item["image"], item["label"]
        img_start_time = time.time()

        img = item["image"] if "image" in item.keys() else item["img"]
        label = item["label"] if "label" in item.keys() else item["fine_label"]
        pixel_values = clip_proc(img).unsqueeze(0).to(DEVICE)
        H = W = 14  # for ViT-B/16 on 224x224 images
        pred_ga, selected = img_to_patch(img, selector, text_features, pixel_values, selection_threshold)
        keep_pct = len(selected)/(H*W)
        total_keep_pct += keep_pct
        correct += (pred_ga == label)
        img_end_time = time.time()

        if save_per_img_metrics:
            inf_time = img_end_time - img_start_time
            memory_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)  # in MB
            records.append({
                'accuracy': (pred_ga == label),
                'inf_time_per_img': inf_time,
                'memory_usage_per_img': memory_usage,
                'keep_tokens': len(selected),
                'keep_pct':  keep_pct*100,
            })
            # Reset CUDA memory tracking for the next image
            torch.cuda.reset_peak_memory_stats(DEVICE)
        if vis:
            # # Save original image
            # img_path = os.path.join(img_dir, f"{idx:05d}_orig.jpg")
            # save_image_copy(img, img_path)

            # # Save heatmap overlay (selected patches = 1.0)
            # heatmap_path = os.path.join(img_dir, f"{idx:05d}_heatmap.png")
            # save_heatmap(img.copy(), selected, (H, W), heatmap_path, alpha=0.5)

            # Save boxes overlay
            boxes_path = os.path.join(img_dir, f"{idx:05d}_boxes.png")
            draw_boxes_on_image_with_preprocess(img.copy(), selected, clip_model, clip_proc, boxes_path)

        idx += 1


    total_time = time.time() - start_time
   
    if save_per_img_metrics:
        per_img_metrics_path = os.path.join(save_dir, "per_img_metrics.jsonl")
        for record in records:
            save_record_jsonl(record, per_img_metrics_path)

    return correct / len(dataset), total_time / len(dataset), total_keep_pct / len(dataset)

def gpp_eval_sigmoid(selector, clip_model, clip_proc, dataset, prompts, text_features, MODEL_ID, DEVICE, save_per_img_metrics, save_dir, vis, selection_threshold):
    total_time, correct = 0.0, 0
    total_keep_pct = 0.0
    start_time = time.time()

    records = []  # To store per-image metrics if needed
    img_dir = os.path.join(save_dir, "images")
    if vis:
        os.makedirs(img_dir, exist_ok=True)
    idx = 0

    for item in dataset:
        # img, label = item["image"], item["label"]
        img_start_time = time.time()
        img = item["image"] if "image" in item.keys() else item["img"]
        label = item["label"] if "label" in item.keys() else item["fine_label"]

        
        pixel_values = clip_proc(img).unsqueeze(0).to(DEVICE)
        H = W = 14  # for ViT-B/16 on 224x224 images
        pred_ga, selected = img_to_patch_with_sigmoid(img, selector, text_features, pixel_values)
        keep_pct = len(selected)/(H*W)
        total_keep_pct += keep_pct
        correct += (pred_ga == label)
        img_end_time = time.time()

        if save_per_img_metrics:
            inf_time = img_end_time - img_start_time
            memory_usage = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)  # in MB
            records.append({
                'accuracy': (pred_ga == label),
                'inf_time_per_img': inf_time,
                'memory_usage_per_img': memory_usage,
                'keep_tokens': len(selected),
                'keep_pct':  keep_pct*100,
            })
            # Reset CUDA memory tracking for the next image
            torch.cuda.reset_peak_memory_stats(DEVICE)

        if vis:
            # # Save original image
            # img_path = os.path.join(img_dir, f"{idx:05d}_orig.jpg")
            # save_image_copy(img, img_path)

            # # Save heatmap overlay (selected patches = 1.0)
            # heatmap_path = os.path.join(img_dir, f"{idx:05d}_heatmap.png")
            # save_heatmap(img.copy(), selected, (H, W), heatmap_path, alpha=0.5)

            # Save boxes overlay
            boxes_path = os.path.join(img_dir, f"{idx:05d}_boxes.png")
            draw_boxes_on_image_with_preprocess(img.copy(), selected, clip_model, clip_proc, boxes_path)

        idx += 1

    
    total_time = time.time() - start_time

    if save_per_img_metrics:
        per_img_metrics_path = os.path.join(save_dir, "per_img_metrics.jsonl")
        for record in records:
            save_record_jsonl(record, per_img_metrics_path)

    return correct / len(dataset), total_time / len(dataset), total_keep_pct / len(dataset)


def gradient_eval(clip_model, clip_proc, dataset, text_features, device: str, keep_pct: float, save_dir: str, vis: bool):
    total_time, correct = 0.0, 0
    total_keep_pct = 0.0

    img_dir = os.path.join(save_dir, str(keep_pct),"images")
    if vis:
        os.makedirs(img_dir, exist_ok=True)

    idx = 0
    for item in dataset:
        img_start_time = time.time()
        img = item["image"] if "image" in item.keys() else item["img"]
        label = item["label"] if "label" in item.keys() else item["fine_label"]

        pixel_values = clip_proc(img).unsqueeze(0).to(device)
        x = clip_model.visual.conv1(pixel_values)
        bsz, dim = x.shape[0], x.shape[1]
        patch_preln = x.reshape(bsz, dim, -1).permute(0, 2, 1)
        num_patches = int(patch_preln.size(1))
        keep = max(1, int(keep_pct * num_patches))

        selected, sel_keep_pct = image_gradient_patch_selection(pixel_values, num_patches, keep=keep)
        img_f = forward_with_selected_patches(clip_model, device, patch_preln, selected)
        probs = (100 * img_f @ text_features.T).softmax(-1)
        pred = probs.argmax().item()

        if vis:
            boxes_path = os.path.join(img_dir, f"{idx:05d}_gradient_boxes.png")
            draw_boxes_on_image_with_preprocess(img.copy(), selected, clip_model, clip_proc, boxes_path)

        total_keep_pct += float(sel_keep_pct)
        correct += int(pred == label)
        total_time += time.time() - img_start_time
        idx += 1

    return correct / len(dataset), total_time / len(dataset), total_keep_pct / len(dataset)

def main():
    # ─── load config.yaml ────────────────────────────────────────────────
    cfg_path = "/home/utn/firi22ka/Desktop/jenga/GeneticPatchPruning/config/benchmark.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples  = cfg["num_samples"]
    dataset_name = cfg["dataset_name"]
    model_id     = cfg["model_id"]
    vis = cfg["visualize"]
    data_split = cfg["split"]
    # loading checkpoint path
    gpp_ckpt = cfg["gpp_model_checkpoint"]
    mlp = cfg["mlp"]
    mixer_mlp_ratio = cfg.get("mixer_mlp_ratio", (0.5, 4.0))
    use_local_data = cfg.get("use_local_data", False)
    data_dir = cfg.get("data_dir", None)
    cache_dir = cfg.get("cache_dir", None)
    selection_threshold = cfg.get("selection_threshold", 0.5)
    keep_pct = float(cfg.get("keep_pct", 0.5))

    # metric flags
    per_img_metrics = cfg.get("per_img_metrics", False)
    save_per_img_metrics = cfg.get("save_per_img_metrics", False)
    save_dir = cfg.get("save_dir", "./benchmark_results")

    # ───────────────────────────────────────────────────────────────────────

    # 1) sampling info
    
    print(f'Evaluating on dataset: {dataset_name} | split: {data_split} ')
    print(f'Model CKpt path: {gpp_ckpt}')
    if num_samples != 0:
        print(f'Evaluating on number of samples: {num_samples}')
    else:
        print(f"Evaluating on full {dataset_name} dataset | split: {data_split}")

    # 2) load data & baseline
    if use_local_data:
        print(f"Loading data from local directory: {data_dir}")
        dataset, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        print(f"Loading data from HuggingFace datasets")   
        dataset, prompts = load_data_normal(dataset_name, num_samples, data_split)
    # orig_acc, orig_time = original_clip(dataset, prompts, model_id, device)

    #loading baseline model
    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float()
    text_features = get_text_features(prompts)


    # 4) run modified versions
    strategies = cfg["strategies"]
    print(f"\n Benchmarking on : {strategies}")
    # assert False
    # keep_pcts  = [0.9, 0.8]  # adjust as desired

    records = []
    for strat in strategies:
        print(f"\n Running strategy: {strat} ")
        if strat == "clip":
            orig_acc, orig_time = original_clip(clip_model, clip_proc, dataset, prompts, text_features, model_id, device)
            records.append({
                'strategy':      strat,
                'keep_pct':      100,
                'accuracy':      orig_acc,
                'avg_time':      orig_time
            })
        elif strat == "gpp":
            model_load_time_start = time.time()
            if mlp == "PatchSelector":
                print(f'Training started using MLP: {mlp}')
                selector = PatchSelector(mixer_mlp_ratio=mixer_mlp_ratio).to(device)
            elif mlp == "PatchSelectorWithSoftmax":
                print(f'Training started using MLP: {mlp}')
                selector = PatchSelectorWithSoftmax().to(device)
            elif mlp == "SimplePatchSelector":
                print(f'Training started using MLP: {mlp}')
                selector = SimplePatchSelector().to(device)
            elif mlp == "SimplePatchSelectorWithDropout":
                print(f'Training started using MLP: {mlp}')
                selector = SimplePatchSelectorWithDropout().to(device)
            elif mlp == "PatchSelectorResBlock":
                print(f'Training started using MLP: {mlp}')
                selector = PatchSelectorResBlock().to(device)
            elif mlp == "LightweightPatchSelector":
                print(f'Training started using MLP: {mlp}')
                selector = LightweightPatchSelector().to(device)
            else: 
                print(f"Please select an MLP in {cfg_path}")
                return

            # selector = PatchSelector().to(device).eval()
            state = torch.load(gpp_ckpt, map_location=device)
            print(f"Loading GPP model from checkpoint: {gpp_ckpt}")
            selector.load_state_dict(state.get("model_state_dict", state))
            selector.eval()
            model_load_time_end = time.time()
            print(f"GPP model loaded in {model_load_time_end - model_load_time_start:.2f} seconds.")

            if mlp == "PatchSelectorWithSoftmax":
                gpp_acc, gpp_time, avg_keep_pct = gpp_eval_sigmoid(selector, clip_model, clip_proc, dataset, prompts, text_features, model_id, device, save_per_img_metrics, save_dir, vis, selection_threshold)
            else:
                gpp_acc, gpp_time, avg_keep_pct = gpp_eval(selector, clip_model, clip_proc, dataset, prompts, text_features, model_id, device, save_per_img_metrics, save_dir, vis, selection_threshold)
                
            print(f'{avg_keep_pct = }')
            records.append({
                'strategy':      strat,
                'keep_pct':  avg_keep_pct*100,
                'accuracy':      gpp_acc,
                'avg_time':      gpp_time
            })
        elif strat == "gradient":
            for keep_pct in [0.9, 0.8, 0.7,0.6, 0.5]:  # adjust as desired
                print(f"\n Evaluating gradient-based selection with keep_pct = {keep_pct} ")
                grad_acc, grad_time, grad_keep_pct = gradient_eval(
                    clip_model,
                    clip_proc,
                    dataset,
                    text_features,
                    device,
                    keep_pct,
                    save_dir,
                    vis,
                )
                records.append({
                    "strategy": strat,
                    "keep_pct": grad_keep_pct * 100,
                    "accuracy": grad_acc,
                    "avg_time": grad_time,
                })
        elif strat == "atk":
            pass  # Placeholder for future implementation

    display_comparison_tables(records, strategies)
            

    

if __name__ == "__main__":
    main()
