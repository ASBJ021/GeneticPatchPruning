import json
import os
from typing import Any, Dict, List, Optional

import torch

from .io import save_record_jsonl, load_completed_indices
from ..model.clip_model import prepare_inputs, forward_with_selected_patches
from .ga import genetic_algorithm, genetic_algorithm_variable_keep


def _to_int_list(values) -> List[int]:
    return [int(v) for v in values]


def _append_jsonl_record(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _save_selected_patch_overlay(img, selected_idx: List[int], processor, save_path: str) -> None:
    """Save selected patches drawn on top of the original image."""
    try:
        from PIL import ImageDraw
        from torchvision.transforms import CenterCrop, Resize

        if not hasattr(img, "copy") or not hasattr(img, "size"):
            return
        if not hasattr(processor, "transforms"):
            return

        resize = next((t for t in processor.transforms if isinstance(t, Resize)), None)
        crop = next((t for t in processor.transforms if isinstance(t, CenterCrop)), None)
        if resize is None or crop is None:
            return

        target = resize.size if isinstance(resize.size, int) else resize.size[0]
        crop_s = crop.size if isinstance(crop.size, int) else crop.size[0]

        overlay = img.copy()
        ow, oh = overlay.size
        if ow < oh:
            nw, nh = target, int(target * oh / ow)
        else:
            nw, nh = int(target * ow / oh), target
        left, top = (nw - crop_s) // 2, (nh - crop_s) // 2
        sx, sy = ow / nw, oh / nh

        grid = crop_s // 16
        draw = ImageDraw.Draw(overlay)
        for ii in selected_idx:
            r, c = divmod(int(ii), grid)
            x0, y0 = (c * 16 + left) * sx, (r * 16 + top) * sy
            draw.rectangle([x0, y0, x0 + 16 * sx, y0 + 16 * sy], outline="green", width=2)
        overlay.save(save_path)
    except Exception as exc:
        print(f"Warning: failed to save GA overlay {save_path}: {exc}")


def _save_original_clip_heatmap(img, x_tokens, model, text_features, original_cls, save_path: str) -> None:
    """Save an original-CLIP patch similarity heatmap for the full-image prediction."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        if not hasattr(img, "copy") or not hasattr(img, "size"):
            return

        probs = (100 * original_cls @ text_features.T).softmax(-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        pred_text_feature = text_features[pred]

        patch_tokens = model.visual.ln_pre(
            x_tokens + model.visual.positional_embedding[1 : x_tokens.size(1) + 1].unsqueeze(0)
        )
        patch_embed = patch_tokens @ model.visual.proj if model.visual.proj is not None else patch_tokens
        patch_embed = patch_embed / patch_embed.norm(dim=-1, keepdim=True)
        sims = torch.matmul(patch_embed[0], pred_text_feature.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()

        num_patches = int(sims.shape[0])
        grid = int(round(num_patches ** 0.5))
        heatmap = sims.reshape(grid, grid)
        rng = float(np.ptp(heatmap))
        norm = (heatmap - heatmap.min()) / (rng + 1e-8)

        base_img = img.copy()
        if base_img.mode != "RGB":
            base_img = base_img.convert("RGB")
        base = np.array(base_img, dtype=np.float32)
        h, w = base.shape[:2]
        hm = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
        colored = plt.get_cmap("jet")(np.array(hm) / 255.0)[..., :3]
        blended = (base * 0.5 + colored * 255 * 0.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(blended).save(save_path)
    except Exception as exc:
        print(f"Warning: failed to save original CLIP heatmap {save_path}: {exc}")


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
    use_clip_labels: bool = False,
    ga_visualization: bool = False,
):
    """Run GA-based patch selection and evaluate CLIP predictions per image."""
    results = []
    ga_vis_root: Optional[str] = None
    if ga_visualization:
        ga_vis_root = os.path.join(os.path.dirname(out_path_jsonl) or ".", "ga_vsialization")
        os.makedirs(ga_vis_root, exist_ok=True)
        print(f"GA visualization logs will be saved to: {ga_vis_root}")

    last_done = load_completed_indices(out_path_jsonl)
    if last_done >= 0:
        print(f"Resuming from image {last_done+1} (last saved was {last_done})")

    for idx, item in enumerate(dataset):
        if idx <= last_done:
            continue  # skip already processed

        print(f"idx = {idx}")
        img, label = item["image"], item["label"]
        print(f'{idx = } , {label = }')
        og_label = int(label)
        pixel_values, text_features, original_cls, patch_tokens = prepare_inputs(
            model, processor, device, img, prompts
        )

        if use_clip_labels:
            # Get CLIP prediction as label
            with torch.no_grad():
                img_feat_full = model.encode_image(pixel_values)
                probs = (100 * img_feat_full @ text_features.T).softmax(-1)
                label = probs.argmax().item()
            print(f'Using CLIP predicted label: {label} while original GT label: {og_label}')


        # Precompute patch token map for pruned forward
        x = model.visual.conv1(processor(img).unsqueeze(0).to(device))
        B, D, N = x.shape[0], x.shape[1], x.numel() // (x.shape[0] * x.shape[1])
        x = x.view(B, D, -1).permute(0, 2, 1)

        num_patches = patch_tokens.size(1)
        keep = max(1, int(keep_pct * num_patches))
        min_keep = max(1, int(min_keep_pct * num_patches))
        max_keep = max(1, int(max_keep_pct * num_patches))
        iteration = 0
        pred = -1
        sel_keep_pct = 0.0
        selected_indices: List[int] = []
        solved = False

        image_ga_dir = None
        ga_steps_path = None
        if ga_visualization and ga_vis_root is not None:
            image_ga_dir = os.path.join(ga_vis_root, f"image_{idx:06d}")
            os.makedirs(image_ga_dir, exist_ok=True)
            ga_steps_path = os.path.join(image_ga_dir, "ga_steps.jsonl")
            orig_clip_heatmap_path = os.path.join(image_ga_dir, "orig_clip_heatmap.png")
            _save_original_clip_heatmap(img, x, model, text_features, original_cls, orig_clip_heatmap_path)
            _append_jsonl_record(
                ga_steps_path,
                {
                    "event": "image_start",
                    "image_id": int(idx),
                    "gt": int(label),
                    "num_patches": int(num_patches),
                    "optimize_keep": bool(optimize_keep),
                    "keep_pct": float(keep_pct),
                    "min_keep_pct": float(min_keep_pct),
                    "max_keep_pct": float(max_keep_pct),
                    "keep_penalty": float(keep_penalty),
                    "orig_clip_heatmap_path": orig_clip_heatmap_path,
                },
            )

        while iteration < 10:
            iter_id = iteration

            def _step_logger(step_payload: Dict[str, Any]) -> None:
                if ga_steps_path is None:
                    return
                payload = {
                    "event": "ga_generation",
                    "image_id": int(idx),
                    "outer_iteration": int(iter_id),
                }
                payload.update(step_payload)
                if "best_kept_indices" in payload:
                    payload["best_kept_indices"] = _to_int_list(payload["best_kept_indices"])
                    if image_ga_dir is not None and ga_visualization:
                        gen_dir = os.path.join(
                            image_ga_dir,
                            "population_generations",
                            f"iter_{iter_id:02d}",
                        )
                        os.makedirs(gen_dir, exist_ok=True)
                        event_tag = payload.get("event", "gen")
                        generation = int(payload.get("generation", -1))
                        gen_overlay_path = os.path.join(
                            gen_dir,
                            f"gen_{generation:03d}_{event_tag}.png",
                        )
                        _save_selected_patch_overlay(
                            img,
                            payload["best_kept_indices"],
                            processor,
                            gen_overlay_path,
                        )
                _append_jsonl_record(ga_steps_path, payload)

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
                    step_callback=_step_logger if ga_visualization else None,
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
                    step_callback=_step_logger if ga_visualization else None,
                )
                sel_keep_pct = len(selected_indices) / num_patches if num_patches > 0 else 0.0

            img_feat = forward_with_selected_patches(model, device, x, selected_indices)
            probs = (100 * img_feat @ text_features.T).softmax(-1)
            # prob1 = torch.argmax(probs, dim=-1).item()
            # prob = probs[0, prob1].item()
            # print(f'{prob = }')
            pred = probs.argmax().item()

            print(f"Iteration {iter_id}: Prediction={pred}, GT={label}, Keep % = {sel_keep_pct}")
            selected_ints = _to_int_list(selected_indices)
            if ga_steps_path is not None:
                _append_jsonl_record(
                    ga_steps_path,
                    {
                        "event": "outer_iteration",
                        "image_id": int(idx),
                        "iteration": int(iter_id),
                        "pred": int(pred),
                        "gt": int(label),
                        "keep_count": int(len(selected_ints)),
                        "keep_pct": float(sel_keep_pct),
                        "selected_indices": selected_ints,
                        "matched_gt": bool(pred == label),
                    },
                )
                if image_ga_dir is not None:
                    overlay_path = os.path.join(
                        image_ga_dir,
                        f"iter_{iter_id:02d}_pred_{int(pred)}_gt_{int(label)}.png",
                    )
                    _save_selected_patch_overlay(img, selected_ints, processor, overlay_path)
            iteration += 1

            if pred == label:
                solved = True
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
                    "selected_indices": selected_ints,
                    "keep_count": len(selected_indices),
                    "keep_pct": sel_keep_pct,
                }
                save_record_jsonl(record, out_path_jsonl)
                print(f"Saved record for image {idx} to {out_path_jsonl}")
                results.append(record)
                break

        if ga_steps_path is not None:
            _append_jsonl_record(
                ga_steps_path,
                {
                    "event": "image_end",
                    "image_id": int(idx),
                    "solved": bool(solved),
                    "final_pred": int(pred),
                    "gt": int(label),
                    "final_keep_count": int(len(selected_indices)),
                    "final_keep_pct": float(sel_keep_pct),
                },
            )

        results.append(
            {
                "image_id": idx,
                "selected_indices": _to_int_list(selected_indices),
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
    ga_visualization: bool = False,
):
    """Run GA-based patch selection and evaluate CLIP predictions per image.

    base_idx sets the global dataset index offset for this rank's slice.
    """
    results = []
    ga_vis_root: Optional[str] = None
    if ga_visualization:
        ga_vis_root = os.path.join(os.path.dirname(out_path_jsonl) or ".", "ga_vsialization")
        os.makedirs(ga_vis_root, exist_ok=True)
        print(f"GA visualization logs will be saved to: {ga_vis_root}")

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
        pred = -1
        sel_keep_pct = 0.0
        selected_indices: List[int] = []
        solved = False

        image_ga_dir = None
        ga_steps_path = None
        if ga_visualization and ga_vis_root is not None:
            image_ga_dir = os.path.join(ga_vis_root, f"image_{global_id:06d}")
            os.makedirs(image_ga_dir, exist_ok=True)
            ga_steps_path = os.path.join(image_ga_dir, "ga_steps.jsonl")
            orig_clip_heatmap_path = os.path.join(image_ga_dir, "orig_clip_heatmap.png")
            _save_original_clip_heatmap(img, x, model, text_features, original_cls, orig_clip_heatmap_path)
            _append_jsonl_record(
                ga_steps_path,
                {
                    "event": "image_start",
                    "image_id": int(global_id),
                    "gt": int(label),
                    "num_patches": int(num_patches),
                    "optimize_keep": bool(optimize_keep),
                    "keep_pct": float(keep_pct),
                    "min_keep_pct": float(min_keep_pct),
                    "max_keep_pct": float(max_keep_pct),
                    "keep_penalty": float(keep_penalty),
                    "orig_clip_heatmap_path": orig_clip_heatmap_path,
                },
            )

        while iteration < 15:
            iter_id = iteration

            def _step_logger(step_payload: Dict[str, Any]) -> None:
                if ga_steps_path is None:
                    return
                payload = {
                    "event": "ga_generation",
                    "image_id": int(global_id),
                    "outer_iteration": int(iter_id),
                }
                payload.update(step_payload)
                if "best_kept_indices" in payload:
                    payload["best_kept_indices"] = _to_int_list(payload["best_kept_indices"])
                    if image_ga_dir is not None and ga_visualization:
                        gen_dir = os.path.join(
                            image_ga_dir,
                            "population_generations",
                            f"iter_{iter_id:02d}",
                        )
                        os.makedirs(gen_dir, exist_ok=True)
                        event_tag = payload.get("event", "gen")
                        generation = int(payload.get("generation", -1))
                        gen_overlay_path = os.path.join(
                            gen_dir,
                            f"gen_{generation:03d}_{event_tag}.png",
                        )
                        _save_selected_patch_overlay(
                            img,
                            payload["best_kept_indices"],
                            processor,
                            gen_overlay_path,
                        )
                _append_jsonl_record(ga_steps_path, payload)

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
                    step_callback=_step_logger if ga_visualization else None,
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
                    step_callback=_step_logger if ga_visualization else None,
                )
                sel_keep_pct = len(selected_indices) / num_patches if num_patches > 0 else 0.0

            img_feat = forward_with_selected_patches(model, device, x, selected_indices)
            probs = (100 * img_feat @ text_features.T).softmax(-1)
            pred = probs.argmax().item()

            print(f"Iteration {iter_id}: Prediction={pred}, GT={label}, Keep % = {sel_keep_pct}")
            selected_ints = _to_int_list(selected_indices)
            if ga_steps_path is not None:
                _append_jsonl_record(
                    ga_steps_path,
                    {
                        "event": "outer_iteration",
                        "image_id": int(global_id),
                        "iteration": int(iter_id),
                        "pred": int(pred),
                        "gt": int(label),
                        "keep_count": int(len(selected_ints)),
                        "keep_pct": float(sel_keep_pct),
                        "selected_indices": selected_ints,
                        "matched_gt": bool(pred == label),
                    },
                )
                if image_ga_dir is not None:
                    overlay_path = os.path.join(
                        image_ga_dir,
                        f"iter_{iter_id:02d}_pred_{int(pred)}_gt_{int(label)}.png",
                    )
                    _save_selected_patch_overlay(img, selected_ints, processor, overlay_path)
            iteration += 1

            if pred == label:
                solved = True
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
                    "selected_indices": selected_ints,
                    "keep_count": len(selected_indices),
                    "keep_pct": sel_keep_pct,
                }
                save_record_jsonl(record, out_path_jsonl)
                print(f"Saved record for image {global_id} to {out_path_jsonl}")
                results.append(record)
                break

        if ga_steps_path is not None:
            _append_jsonl_record(
                ga_steps_path,
                {
                    "event": "image_end",
                    "image_id": int(global_id),
                    "solved": bool(solved),
                    "final_pred": int(pred),
                    "gt": int(label),
                    "final_keep_count": int(len(selected_indices)),
                    "final_keep_pct": float(sel_keep_pct),
                },
            )

        results.append(
            {
                "image_id": global_id,
                "selected_indices": _to_int_list(selected_indices),
                "keep_count": len(selected_indices),
                "keep_pct": sel_keep_pct,
            }
        )

    return results
