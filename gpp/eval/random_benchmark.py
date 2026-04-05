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
from gpp.eval.benchmark import gpp_eval, gpp_eval_sigmoid, original_clip
from torch.nn.functional import normalize
from gpp.genetic_algo.image_gradent_selection import image_gradient_patch_selection


import clip
import time, math

def modified_clip_dropout(dataset, prompts,  MODEL_ID, DEVICE, keep_pct=0.5, strategy="random", seed=42, visualize=False):
    # torch.manual_seed(seed)
    model, proc = clip.load(MODEL_ID, DEVICE); model = model.float()
    # precompute text embeddings
    toks = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        txt_feats = model.encode_text(toks)
        txt_feats /= txt_feats.norm(dim=-1,keepdim=True)

    
    total, correct = 0.0, 0
    for item in dataset:
        img, label = item["image"], item["label"]
        feat_tgt = txt_feats[label:label+1]
        img_input = proc(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            start = time.time()
            # extract patch tokens
            x = model.visual.conv1(img_input)
            B,D,N = x.shape[0], x.shape[1], x.numel()//(x.shape[0]*x.shape[1])
            x = x.reshape(B, D, -1).permute(0,2,1)
            keep = max(1, int(keep_pct * x.shape[1]))

            # choose indices
            if strategy=="random":
                idx = torch.randperm(x.shape[1], device=DEVICE)[:keep]
            elif strategy=="uniform":
                g = int(math.sqrt(x.shape[1])); step = max(1, g//int(math.sqrt(keep)))
                coords = [(i,j) for i in range(0,g,step) for j in range(0,g,step)]
                idx = torch.tensor([i*g+j for i,j in coords],device=DEVICE)[:keep]
            elif strategy=="similarity":
                pos = model.visual.positional_embedding[1:N+1].unsqueeze(0)
                x_pe = model.visual.ln_pre(x + pos)
                patch_e = x_pe @ model.visual.proj if model.visual.proj is not None else x_pe
                sims = normalize(patch_e.squeeze(0),dim=-1) @ normalize(feat_tgt.squeeze(0),dim=-1)
                idx = sims.topk(keep).indices
                # plot_heatmap_overlay(img, sims.cpu().numpy(), (g,g), alpha=0.4)
            elif strategy=="gradient":
                idx_list, _ = image_gradient_patch_selection(img_input, x.shape[1], keep=keep)
                idx = torch.tensor(idx_list, device=DEVICE, dtype=torch.long)
            else:
                raise ValueError(f"Unknown strategy {strategy}")
            idx = torch.sort(idx)[0]

            
            # rebuild sequence & forward
            cls = model.visual.class_embedding + torch.zeros(1,1,D,device=DEVICE)
            seq_all = torch.cat([cls, x], dim=1)
            pos_all = model.visual.positional_embedding[:seq_all.size(1)].unsqueeze(0)
            seq_all = model.visual.ln_pre(seq_all + pos_all)
            keep_idx = torch.cat([torch.zeros(1,dtype=torch.long,device=DEVICE), idx+1])
            seq = seq_all[:, keep_idx, :]
            # learnable_img_tokens = ''

            
            z = model.visual.transformer(seq.permute(1,0,2))
            z = model.visual.ln_post(z.permute(1,0,2)[:,0])

            img_f = (z @ model.visual.proj) if model.visual.proj is not None else z
            img_f /= img_f.norm(dim=-1,keepdim=True)
            sim2 = (100*img_f @ txt_feats.T).softmax(-1)
            pred = sim2.argmax().item()
            total += time.time() - start
            correct += (pred==label)
            # optional: viz patches
            # if visualize:
                # patches = patchify(img, resolution=224, patch_size=16)
                # viz_patches(patches, topk=idx.cpu(), img_title=f"{strategy}_{keep_pct}_{label}")
    return correct/len(dataset), total/len(dataset)

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
    keep_pcts  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1] 

    records = []
    for strat in strategies:
        for pct in keep_pcts:
            print(f"\nEvaluating strategy: {strat} | keep_pct: {pct}")
            acc, avg_time = modified_clip_dropout(
                dataset, prompts,
                model_id,
                device,
                keep_pct=pct,
                strategy=strat,
                seed=42,
                visualize=vis
            )
            # record both generic and strat‐specific fields
            records.append({
                'strategy':      strat,
                'keep_pct':      pct,
                'accuracy':      acc,
                'avg_time':      avg_time,
                f'{strat}_acc':       acc,
                f'{strat}_avg_time':  avg_time,
            })

    display_comparison_tables(records, strategies)
            

    

if __name__ == "__main__":
    main()
