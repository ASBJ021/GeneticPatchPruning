from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clip
import torch
import yaml

try:
    from datasets import get_dataset_split_names
except Exception:
    get_dataset_split_names = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpp.dataset.data_utils import load_data_folder, load_data_normal
from gpp.eval.eval_ga_distill import (
    build_selector,
    get_text_features,
    gpp_eval_adaptive_topk,
    original_clip,
)
from gpp.genetic_algo.io import save_record_jsonl


HF_DATASET_ALIASES: Dict[str, str] = {
    "caltech101": "flwrlabs/caltech101",
    "oxfordpets": "timm/oxford-iiit-pet",
    "flowers102": "nkirschi/oxford-flowers",
    "food101": "ethz/food101",
    "fgvcaircraft": "Donghyun99/FGVC-Aircraft",
    "dtd": "tanganke/dtd",
    "ucf101": "flwrlabs/ucf101",
    "imagenet": "ILSVRC/imagenet-1k",
    "imagenet1k": "ILSVRC/imagenet-1k",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_hf_dataset_name(name: str) -> Tuple[str, Optional[str]]:
    key = _normalize_name(name)
    resolved = HF_DATASET_ALIASES.get(key)
    if resolved:
        return resolved, f"HF alias remap: '{name}' -> '{resolved}'"
    return name, None


def _resolve_datasets(cfg: Dict[str, Any], cli_datasets: Optional[List[str]]) -> List[str]:
    if cli_datasets:
        return cli_datasets

    ds_cfg = cfg.get("datasets")
    if isinstance(ds_cfg, list) and ds_cfg:
        return [str(x) for x in ds_cfg]
    if isinstance(ds_cfg, str) and ds_cfg.strip():
        return [x.strip() for x in ds_cfg.split(",") if x.strip()]

    return [str(cfg["dataset_name"])]


def _resolve_split(cfg: Dict[str, Any], dataset_name: str, split_override: Optional[str]) -> str:
    if split_override:
        return split_override

    ds_splits = cfg.get("dataset_splits", {})
    if isinstance(ds_splits, dict) and dataset_name in ds_splits:
        return str(ds_splits[dataset_name])

    return str(cfg.get("split", "test"))


def _resolve_hf_split(dataset_name: str, requested_split: str) -> Tuple[str, Optional[str]]:
    if not requested_split or get_dataset_split_names is None:
        return requested_split, None

    try:
        available = get_dataset_split_names(dataset_name)
    except Exception:
        return requested_split, None

    if not available:
        return requested_split, None

    by_lower = {s.lower(): s for s in available}
    alias = {"val": "validation", "valid": "validation", "dev": "validation"}

    req_lower = requested_split.lower()
    req_norm = alias.get(req_lower, req_lower)

    if req_norm in by_lower:
        chosen = by_lower[req_norm]
        if chosen != requested_split:
            msg = (
                f"Split remap for dataset={dataset_name}: requested='{requested_split}' "
                f"-> using='{chosen}' (available={available})"
            )
            return chosen, msg
        return chosen, None

    preferences = {
        "test": ["test", "validation", "valid", "val", "train"],
        "validation": ["validation", "valid", "val", "test", "train"],
        "val": ["validation", "valid", "val", "test", "train"],
        "train": ["train", "validation", "valid", "val", "test"],
    }

    for cand in preferences.get(req_norm, [req_norm, "test", "validation", "train"]):
        cand_norm = alias.get(cand, cand)
        if cand_norm in by_lower:
            chosen = by_lower[cand_norm]
            msg = (
                f"Split fallback for dataset={dataset_name}: requested='{requested_split}' "
                f"-> using='{chosen}' (available={available})"
            )
            return chosen, msg

    return requested_split, None


def _resolve_data_dir(cfg: Dict[str, Any], dataset_name: str) -> Optional[str]:
    data_dirs = cfg.get("data_dirs", {})
    if isinstance(data_dirs, dict) and dataset_name in data_dirs:
        return str(data_dirs[dataset_name])

    template = cfg.get("data_dir_template")
    if isinstance(template, str) and template:
        return template.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))

    data_dir = cfg.get("data_dir")
    return str(data_dir) if data_dir else None


def _print_dataset_summary_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        return

    headers = [
        "dataset",
        "keep_pct (%)",
        "gpp acc (%)",
        "clip_acc (%)",
        "gpp_inf_time",
        "clip_inf_time",
    ]

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def _line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (widths[h] + 2) for h in headers) + "+"

    def _fmt(row: Dict[str, str]) -> str:
        return "| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |"

    print("\nSummary Table")
    print(_line("-"))
    print(_fmt({h: h for h in headers}))
    print(_line("="))
    for row in rows:
        print(_fmt(row))
    print(_line("-"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GA distill benchmark on multiple datasets and append summaries to JSONL."
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "benchmark_ga_distill.yaml"))
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names. Overrides config datasets/dataset_name.")
    parser.add_argument("--split", type=str, default=None, help="Split override for all datasets.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples per dataset (0 = full split).")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path. Default: <save_dir>/multi_dataset_benchmark.jsonl")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately if a dataset fails.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = str(cfg.get("device", "cuda"))
    if not torch.cuda.is_available():
        device = "cpu"

    model_id = str(cfg["model_id"])
    strategies = list(cfg.get("strategies", ["gpp"]))
    if "clip" not in strategies:
        strategies = ["clip"] + strategies

    save_per_img_metrics = bool(cfg.get("save_per_img_metrics", False))
    visualize = bool(cfg.get("visualize", False))
    use_local_data = bool(cfg.get("use_local_data", False))
    cache_dir = cfg.get("cache_dir")
    base_save_dir = str(cfg.get("save_dir", "./benchmark_results"))
    num_samples = int(args.num_samples if args.num_samples is not None else int(cfg.get("num_samples", 0)))
    out_path = args.output or os.path.join(base_save_dir, "multi_dataset_benchmark.jsonl")
    datasets = _resolve_datasets(cfg, args.datasets)

    selection_cfg = cfg.get("selection", {})
    min_keep_tokens = int(selection_cfg.get("min_keep_tokens", cfg.get("min_keep_tokens", 1)))
    max_keep_tokens = int(selection_cfg.get("max_keep_tokens", cfg.get("max_keep_tokens", 0)))

    print(f"Running multi-dataset benchmark on: {datasets}")
    print(f"Output JSONL: {out_path}")
    print(f"Device: {device} | model: {model_id} | strategies: {strategies}")

    clip_model, clip_proc = clip.load(model_id, device)
    clip_model = clip_model.float().eval()

    selector = None
    outputs_are_probs = False
    if "gpp" in strategies:
        mlp = str(cfg["mlp"])
        mixer_mlp_ratio = cfg.get("mixer_mlp_ratio", None)
        gpp_ckpt = str(cfg["gpp_model_checkpoint"])
        selector, outputs_are_probs = build_selector(mlp, device, mixer_mlp_ratio)
        state = torch.load(gpp_ckpt, map_location=device)
        selector.load_state_dict(state.get("model_state_dict", state))
        selector.eval()
        print(f"Loaded selector checkpoint: {gpp_ckpt}")

    dataset_summary: Dict[str, Dict[str, str]] = {}

    for dataset_name in datasets:
        requested_split = _resolve_split(cfg, dataset_name, args.split)
        split = requested_split
        split_note = None

        hf_dataset_name = dataset_name
        alias_note = None
        if not use_local_data:
            hf_dataset_name, alias_note = _resolve_hf_dataset_name(dataset_name)
            split, split_note = _resolve_hf_split(hf_dataset_name, requested_split)

        dataset_slug = dataset_name.replace("/", "-")
        dataset_save_dir = os.path.join(base_save_dir, dataset_slug)
        start_ts = _utc_now_iso()

        dataset_summary[dataset_name] = {
            "dataset": dataset_name,
            "keep_pct (%)": "-",
            "gpp acc (%)": "-",
            "clip_acc (%)": "-",
            "gpp_inf_time": "-",
            "clip_inf_time": "-",
        }

        print(f"\nEvaluating dataset={dataset_name} (hf={hf_dataset_name}) split={split} num_samples={num_samples}")
        if alias_note:
            print(alias_note)
        if split_note:
            print(split_note)

        try:
            if use_local_data:
                data_dir = _resolve_data_dir(cfg, dataset_name)
                if not data_dir:
                    raise ValueError(
                        "use_local_data=true but no data_dir/data_dir_template/data_dirs entry found for dataset."
                    )
                dataset, prompts = load_data_folder(data_dir, num_samples, SPLIT=split, cache_dir=cache_dir)
            else:
                dataset, prompts = load_data_normal(hf_dataset_name, num_samples, split)

            text_features = get_text_features(prompts, clip_model, device)

            if "clip" in strategies:
                clip_acc, clip_time, _ = original_clip(clip_model, clip_proc, dataset, text_features, device)
                clip_record = {
                    "status": "ok",
                    "dataset": dataset_name,
                    "hf_dataset": hf_dataset_name,
                    "split": split,
                    "num_samples": int(len(dataset)),
                    "requested_num_samples": num_samples,
                    "strategy": "clip",
                    "accuracy": float(clip_acc),
                    "inf_time": float(clip_time),
                    "keep_pct": 100.0,
                    "model_id": model_id,
                }
                save_record_jsonl(clip_record, out_path)
                dataset_summary[dataset_name]["clip_acc (%)"] = f"{clip_acc * 100.0:.2f}"
                dataset_summary[dataset_name]["clip_inf_time"] = f"{clip_time:.4f}"

            if "gpp" in strategies:
                if selector is None:
                    raise RuntimeError("`gpp` strategy requested but selector is not initialized.")

                gpp_acc, gpp_time, keep_mean, _, _, _ = gpp_eval_adaptive_topk(
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
                    save_dir=dataset_save_dir,
                    visualize=visualize,
                    annotation_map=None,
                )

                gpp_record = {
                    "status": "ok",
                    "dataset": dataset_name,
                    "hf_dataset": hf_dataset_name,
                    "split": split,
                    "num_samples": int(len(dataset)),
                    "requested_num_samples": num_samples,
                    "strategy": "gpp_adaptive_topk",
                    "accuracy": float(gpp_acc),
                    "inf_time": float(gpp_time),
                    "keep_pct": float(keep_mean * 100.0),
                    "model_id": model_id,
                }
                save_record_jsonl(gpp_record, out_path)
                dataset_summary[dataset_name]["gpp acc (%)"] = f"{gpp_acc * 100.0:.2f}"
                dataset_summary[dataset_name]["gpp_inf_time"] = f"{gpp_time:.4f}"
                dataset_summary[dataset_name]["keep_pct (%)"] = f"{keep_mean * 100.0:.2f}"

            print(f"Finished dataset={dataset_name} in {start_ts} -> {_utc_now_iso()}")

        except Exception as exc:
            error_record = {
                "status": "error",
                "dataset": dataset_name,
                "hf_dataset": hf_dataset_name,
                "split": split,
                "requested_split": requested_split,
                "requested_num_samples": num_samples,
                "error": f"{type(exc).__name__}: {exc}",
                "device": device,
                "model_id": model_id,
            }
            save_record_jsonl(error_record, out_path)
            print(f"Failed dataset={dataset_name}: {error_record['error']}")
            if args.stop_on_error:
                raise

    _print_dataset_summary_table([dataset_summary[d] for d in datasets])
    print(f"\nDone. Summary JSONL saved to: {out_path}")


if __name__ == "__main__":
    main()
