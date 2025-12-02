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

from gpp.dataset.data_utils import load_data_normal
# from gpp.model.clip_model import original_clip
from gpp.eval.compare import display_comparison_tables
from gpp.model.model import PatchSelector, images_to_patches
from gpp.model.clip_model import forward_with_selected_patches, load_clip
from gpp.eval.evaluate import img_to_patch, get_image_feature_full, get_text_features

import clip
import time

def original_clip(model, proc, dataset, prompts, txt_f ,MODEL_ID, DEVICE):
    # model, proc = clip.load(MODEL_ID, DEVICE)
    # model = model.float()
    total, correct = 0.0, 0
    start = time.time()
    for item in dataset:
        img, label = item["image"], item["label"]
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

def gpp_eval(selector, clip_model, clip_proc, dataset, prompts, text_features, MODEL_ID, DEVICE):
    total_time, correct = 0.0, 0
    start_time = time.time()
    for item in dataset:
        img, label = item["image"], item["label"]
        pixel_values = clip_proc(img).unsqueeze(0).to(DEVICE)
        H = W = 14  # for ViT-B/16 on 224x224 images
        pred_ga, selected = img_to_patch(img, selector, text_features, pixel_values)
        correct += (pred_ga == label)
    total_time = time.time() - start_time
    return correct / len(dataset), total_time / len(dataset)

def main():
    # ─── load config.yaml ────────────────────────────────────────────────
    # cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
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
    # ───────────────────────────────────────────────────────────────────────

    # 1) sampling info
    
    print(f'Evaluating on dataset: {dataset_name} | split: {data_split} ')
    if num_samples != 0:
        print(f'Evaluating on number of samples: {num_samples}')
    else:
        print(f"Evaluating on full {dataset_name} dataset")

    # 2) load data & baseline
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
        if strat == "clip":
            orig_acc, orig_time = original_clip(clip_model, clip_proc, dataset, prompts, text_features, model_id, device)
            records.append({
                'strategy':      strat,
                'keep_pct':      1.0,
                'accuracy':      orig_acc,
                'avg_time':      orig_time
            })
        elif strat == "gpp":
            selector = PatchSelector().to(device).eval()
            state = torch.load(gpp_ckpt, map_location=device)
            print(f"Loading GPP model from checkpoint: {gpp_ckpt}")
            selector.load_state_dict(state.get("model_state_dict", state))
            selector.eval()
            gpp_acc, gpp_time = gpp_eval(selector, clip_model, clip_proc, dataset, prompts, text_features, model_id, device)
            records.append({
                'strategy':      strat,
                'keep_pct':      1.0,
                'accuracy':      gpp_acc,
                'avg_time':      gpp_time
            })
        elif strat == "atk":
            pass  # Placeholder for future implementation

    display_comparison_tables(records, strategies)
            

    

if __name__ == "__main__":
    main()