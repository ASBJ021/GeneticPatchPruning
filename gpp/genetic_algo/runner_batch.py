"""
Batch processing version of patch_modified_clip using multiprocessing.
Processes multiple images in parallel using worker pool.
"""

from typing import List
import torch
import multiprocessing as mp
from multiprocessing import Pool
import os
from functools import partial

from .io import save_record_jsonl, load_completed_indices
from ..model.clip_model import prepare_inputs, forward_with_selected_patches
from .ga import genetic_algorithm, genetic_algorithm_variable_keep


def process_single_image(
    item_data,
    idx: int,
    prompts: List[str],
    model,
    processor,
    device: str,
    keep_pct: float,
    viz: bool = False,
    patchify_fn=None,
    viz_patches_fn=None,
    optimize_keep: bool = True,
    min_keep_pct: float = 0.1,
    max_keep_pct: float = 0.9,
    keep_penalty: float = 0.1,
):
    """
    Process a single image and return results.
    This runs in a separate process.
    """
    try:
        img, label = item_data["image"], item_data["label"]
        print(f'Processing idx = {idx}, label = {label}')
        
        # Prepare inputs
        pixel_values, text_features, original_cls, patch_tokens = prepare_inputs(
            model, processor, device, img, prompts
        )

        # Precompute patch token map for pruned forward
        x = model.visual.conv1(processor(img).unsqueeze(0).to(device))
        B, D, N = x.shape[0], x.shape[1], x.numel() // (x.shape[0] * x.shape[1])
        x = x.view(B, D, -1).permute(0, 2, 1)

        num_patches = patch_tokens.size(1)
        keep = max(1, int(keep_pct * num_patches))
        min_keep = max(1, int(min_keep_pct * num_patches))
        max_keep = max(1, int(max_keep_pct * num_patches))
        iteration = 0
        selected_indices = None
        sel_keep_pct = 0

        while iteration < 10:
            if optimize_keep:
                selected_indices, sel_keep_pct = genetic_algorithm_variable_keep(
                    model,
                    device,
                    x,
                    text_features,
                    original_cls,
                    num_patches,
                    min_keep,
                    max_keep,
                    keep_penalty=keep_penalty,
                )
            else:
                selected_indices = genetic_algorithm(
                    model,
                    device,
                    x,
                    text_features,
                    original_cls,
                    num_patches,
                    keep,
                )
                sel_keep_pct = len(selected_indices) / num_patches if num_patches > 0 else 0.0

            img_feat = forward_with_selected_patches(model, device, x, selected_indices)
            probs = (100 * img_feat @ text_features.T).softmax(-1)
            pred = probs.argmax().item()

            print(f"  Iteration {iteration}: Prediction={pred}, GT={label}, Keep % = {sel_keep_pct}")
            iteration += 1

            if pred == label:
                if viz and patchify_fn is not None and viz_patches_fn is not None:
                    patches = patchify_fn(img, resolution=224, patch_size=16)
                    viz_patches_fn(
                        patches,
                        topk=selected_indices,
                        img_title=f"best_patches_{idx}_pred={pred}_label={label}",
                    )
                break

        # Return result
        record = {
            "image_id": idx,
            "pred": int(pred),
            "gt": int(label),
            "selected_indices": selected_indices,
            "keep_count": len(selected_indices) if selected_indices else 0,
            "keep_pct": sel_keep_pct,
        }
        return record, None
        
    except Exception as e:
        print(f"Error processing image {idx}: {str(e)}")
        return None, str(e)


def patch_modified_clip_batch(
    dataset,
    prompts: List[str],
    model,
    processor,
    device: str,
    keep_pct: float,
    out_path_jsonl: str,
    viz: bool = False,
    patchify_fn=None,
    viz_patches_fn=None,
    optimize_keep: bool = True,
    min_keep_pct: float = 0.1,
    max_keep_pct: float = 0.9,
    keep_penalty: float = 0.1,
    batch_size: int = 4,
    num_workers: int = None,
):
    """
    Run GA-based patch selection with batch processing.
    
    Args:
        dataset: Dataset to process
        batch_size: Number of images to process per batch
        num_workers: Number of parallel workers (default: CPU count / 2)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} workers for batch processing")
    
    results = []
    last_done = load_completed_indices(out_path_jsonl)
    if last_done >= 0:
        print(f"Resuming from image {last_done+1} (last saved was {last_done})")

    # Create batches
    batch_items = []
    batch_indices = []
    
    for idx, item in enumerate(dataset):
        if idx <= last_done:
            continue
        
        batch_items.append(item)
        batch_indices.append(idx)
        
        # Process batch when full or at end of dataset
        if len(batch_items) == batch_size:
            print(f"\nProcessing batch: images {batch_indices[0]} to {batch_indices[-1]}")
            results.extend(
                _process_batch(
                    batch_items,
                    batch_indices,
                    prompts,
                    model,
                    processor,
                    device,
                    keep_pct,
                    out_path_jsonl,
                    viz,
                    patchify_fn,
                    viz_patches_fn,
                    optimize_keep,
                    min_keep_pct,
                    max_keep_pct,
                    keep_penalty,
                )
            )
            batch_items = []
            batch_indices = []
    
    # Process remaining batch
    if batch_items:
        print(f"\nProcessing final batch: images {batch_indices[0]} to {batch_indices[-1]}")
        results.extend(
            _process_batch(
                batch_items,
                batch_indices,
                prompts,
                model,
                processor,
                device,
                keep_pct,
                out_path_jsonl,
                viz,
                patchify_fn,
                viz_patches_fn,
                optimize_keep,
                min_keep_pct,
                max_keep_pct,
                keep_penalty,
            )
        )

    return results


def _process_batch(
    batch_items,
    batch_indices,
    prompts,
    model,
    processor,
    device,
    keep_pct,
    out_path_jsonl,
    viz,
    patchify_fn,
    viz_patches_fn,
    optimize_keep,
    min_keep_pct,
    max_keep_pct,
    keep_penalty,
):
    """
    Process a single batch of images sequentially.
    (True parallelization requires model to be picklable, which is complex)
    """
    results = []
    
    for item, idx in zip(batch_items, batch_indices):
        record, error = process_single_image(
            item,
            idx,
            prompts,
            model,
            processor,
            device,
            keep_pct,
            viz,
            patchify_fn,
            viz_patches_fn,
            optimize_keep,
            min_keep_pct,
            max_keep_pct,
            keep_penalty,
        )
        
        if record is not None:
            save_record_jsonl(record, out_path_jsonl)
            print(f"✓ Saved record for image {idx} to {out_path_jsonl}")
            results.append(record)
        else:
            print(f"✗ Error processing image {idx}: {error}")
    
    return results
