from typing import List
import torch

from .io import save_record_jsonl, load_completed_indices
from ..model.clip_model import prepare_inputs, forward_with_selected_patches
from .ga import genetic_algorithm, genetic_algorithm_variable_keep


def patch_modified_clip(
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
):
    """Run GA-based patch selection and evaluate CLIP predictions per image."""
    results = []
    last_done = load_completed_indices(out_path_jsonl)
    if last_done >= 0:
        print(f"Resuming from image {last_done+1} (last saved was {last_done})")

    for idx, item in enumerate(dataset):
        if idx <= last_done:
            continue  # skip already processed

        print(f"idx = {idx}")
        img, label = item["image"], item["label"]
        print(f'{idx = } , {label = }')
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
            # prob1 = torch.argmax(probs, dim=-1).item()
            # prob = probs[0, prob1].item()
            # print(f'{prob = }')
            pred = probs.argmax().item()

            print(f"Iteration {iteration}: Prediction={pred}, GT={label}, Keep % = {sel_keep_pct}")
            iteration += 1

            if pred == label:
                if viz and patchify_fn is not None and viz_patches_fn is not None:
                    patches = patchify_fn(img, resolution=224, patch_size=16)
                    viz_patches_fn(
                        patches,
                        topk=selected_indices,
                        img_title=f"best_patches_{idx}_pred={pred}_label={label}",
                    )

                record = {
                    "image_id": idx,
                    "pred": int(pred),
                    "gt": int(label),
                    "selected_indices": selected_indices,
                    "keep_count": len(selected_indices),
                    "keep_pct": sel_keep_pct,
                }
                save_record_jsonl(record, out_path_jsonl)
                print(f"Saved record for image {idx} to {out_path_jsonl}")
                results.append(record)
                break

        results.append(
            {
                "image_id": idx,
                "selected_indices": selected_indices,
                "keep_count": len(selected_indices),
                "keep_pct": sel_keep_pct,
            }
        )

    return results


def parallel_patch_modified_clip(
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
    base_idx: int = 0,
):
    """Run GA-based patch selection and evaluate CLIP predictions per image.

    base_idx sets the global dataset index offset for this rank's slice.
    """
    results = []
    last_done = load_completed_indices(out_path_jsonl)
    if last_done >= 0:
        print(f"Resuming from image {last_done+1} (last saved was {last_done})")

    for idx, item in enumerate(dataset):
        global_id = base_idx + idx
        if global_id <= last_done:
            continue  # skip already processed

        print(f"idx = {idx} (global_id={global_id})")
        img, label = item["image"], item["label"]
        print(f'{idx = } , {label = }')

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

        while iteration < 15:
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

            print(f"Iteration {iteration}: Prediction={pred}, GT={label}, Keep % = {sel_keep_pct}")
            iteration += 1

            if pred == label:
                if viz and patchify_fn is not None and viz_patches_fn is not None:
                    patches = patchify_fn(img, resolution=224, patch_size=16)
                    viz_patches_fn(
                        patches,
                        topk=selected_indices,
                        img_title=f"best_patches_{global_id}_pred={pred}_label={label}",
                    )

                record = {
                    "image_id": global_id,
                    "pred": int(pred),
                    "gt": int(label),
                    "selected_indices": selected_indices,
                    "keep_count": len(selected_indices),
                    "keep_pct": sel_keep_pct,
                }
                save_record_jsonl(record, out_path_jsonl)
                print(f"Saved record for image {global_id} to {out_path_jsonl}")
                results.append(record)
                break

        results.append(
            {
                "image_id": global_id,
                "selected_indices": selected_indices,
                "keep_count": len(selected_indices),
                "keep_pct": sel_keep_pct,
            }
        )

    return results
