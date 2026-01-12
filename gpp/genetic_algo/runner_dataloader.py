"""
Advanced batch processing with PyTorch DataLoader for memory efficiency.
Better for very large datasets.
"""

from typing import List
import torch
from torch.utils.data import DataLoader
import json

from .io import save_record_jsonl, load_completed_indices
from ..model.clip_model import prepare_inputs, forward_with_selected_patches
from .ga import genetic_algorithm, genetic_algorithm_variable_keep


def patch_modified_clip_dataloader(
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
    num_workers: int = 0,
):
    """
    Process dataset using PyTorch DataLoader for batching.
    
    Key advantage: DataLoader handles memory efficiently with num_workers
    for background data loading while GPU processes.
    
    Args:
        batch_size: Images per batch (still process individually through GA,
                   but load batches efficiently)
        num_workers: Parallel processes for data loading (0=main process only)
    """
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    
    results = []
    last_done = load_completed_indices(out_path_jsonl)
    global_idx = 0
    
    if last_done >= 0:
        print(f"Resuming from image {last_done+1} (last saved was {last_done})")

    # Process batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1} - Processing {len(batch['image'])} images")
        print(f"{'='*60}")
        
        for i in range(len(batch['image'])):
            if global_idx <= last_done:
                global_idx += 1
                continue
                
            img = batch['image'][i]
            label = batch['label'][i].item() if isinstance(batch['label'][i], torch.Tensor) else batch['label'][i]
            
            print(f"\n  Image {global_idx}: Label={label}")
            
            try:
                # Prepare inputs
                pixel_values, text_features, original_cls, patch_tokens = prepare_inputs(
                    model, processor, device, img, prompts
                )

                # Precompute patch token map
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

                # GA loop
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

                    print(f"    Iter {iteration}: Pred={pred}, GT={label}, Keep%={sel_keep_pct:.2f}")
                    iteration += 1

                    if pred == label:
                        if viz and patchify_fn is not None and viz_patches_fn is not None:
                            patches = patchify_fn(img, resolution=224, patch_size=16)
                            viz_patches_fn(
                                patches,
                                topk=selected_indices,
                                img_title=f"best_patches_{global_idx}_pred={pred}_label={label}",
                            )
                        break

                # Save result
                record = {
                    "image_id": global_idx,
                    "pred": int(pred),
                    "gt": int(label),
                    "selected_indices": selected_indices,
                    "keep_count": len(selected_indices) if selected_indices else 0,
                    "keep_pct": float(sel_keep_pct),
                }
                save_record_jsonl(record, out_path_jsonl)
                print(f"  ✓ Saved image {global_idx}")
                results.append(record)
                
            except Exception as e:
                print(f"  ✗ Error processing image {global_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            global_idx += 1

    return results
