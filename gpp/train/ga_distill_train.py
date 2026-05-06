from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpp.dataset.data_utils import PatchIndexDataset, load_data_folder, load_data_normal, split_dataset
from gpp.model.model import (
    LightweightPatchSelector,
    PatchSelector,
    PatchSelectorResBlock,
    PatchSelectorWithSoftmax,
    SimplePatchSelector,
    SimplePatchSelectorWithDropout,
)

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _parse_placeholder_tokens(expr: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"([^\.\[\]]+)|\[(.*?)\]", expr):
        token = match.group(1) if match.group(1) is not None else match.group(2)
        if token is None:
            continue
        token = token.strip().strip("'\"")
        if token:
            tokens.append(token)
    if not tokens:
        raise ValueError(f"Invalid placeholder expression: {expr}")
    return tokens


def _lookup_cfg_value(cfg_obj, expr: str):
    current = cfg_obj
    for token in _parse_placeholder_tokens(expr):
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
            continue
        if isinstance(current, list):
            index = int(token)
            current = current[index]
            continue
        raise KeyError(token)
    return current


def _resolve_placeholders_once(obj, cfg_obj) -> tuple[object, bool]:
    if isinstance(obj, dict):
        changed = False
        resolved_dict = {}
        for key, value in obj.items():
            resolved_value, value_changed = _resolve_placeholders_once(value, cfg_obj)
            resolved_dict[key] = resolved_value
            changed = changed or value_changed
        return resolved_dict, changed

    if isinstance(obj, list):
        changed = False
        resolved_list = []
        for value in obj:
            resolved_value, value_changed = _resolve_placeholders_once(value, cfg_obj)
            resolved_list.append(resolved_value)
            changed = changed or value_changed
        return resolved_list, changed

    if isinstance(obj, str):
        def replace_match(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            value = _lookup_cfg_value(cfg_obj, expr)
            return str(value)

        resolved_str = PLACEHOLDER_PATTERN.sub(replace_match, obj)
        return resolved_str, resolved_str != obj

    return obj, False


def _contains_placeholder(obj) -> bool:
    if isinstance(obj, dict):
        return any(_contains_placeholder(value) for value in obj.values())
    if isinstance(obj, list):
        return any(_contains_placeholder(value) for value in obj)
    if isinstance(obj, str):
        return bool(PLACEHOLDER_PATTERN.search(obj))
    return False


def resolve_config_placeholders(cfg: dict, max_passes: int = 5) -> dict:
    resolved = deepcopy(cfg)
    for _ in range(max_passes):
        resolved, changed = _resolve_placeholders_once(resolved, resolved)
        if not changed:
            return resolved
    if _contains_placeholder(resolved):
        raise ValueError("Unable to fully resolve config placeholders.")
    return resolved


def log_message(message: str, log_path: Optional[str] = None) -> None:
    timestamped_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped_message)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(timestamped_message + "\n")


def _clip_preprocess_batch(
    imgs: torch.Tensor,
    input_resolution: int,
    clip_mean: torch.Tensor,
    clip_std: torch.Tensor,
) -> torch.Tensor:
    if imgs.shape[-2:] != (input_resolution, input_resolution):
        imgs = F.interpolate(
            imgs,
            size=(input_resolution, input_resolution),
            mode="bilinear",
            align_corners=False,
        )
    return (imgs - clip_mean) / clip_std


@torch.no_grad()
def extract_patch_and_cls_tokens_batch(
    clip_model,
    imgs: torch.Tensor,
    input_resolution: int,
    clip_mean: torch.Tensor,
    clip_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = _clip_preprocess_batch(imgs, input_resolution, clip_mean, clip_std)

    x = clip_model.visual.conv1(pixel_values)
    bsz, dim, _, _ = x.shape
    tokens = x.reshape(bsz, dim, -1).permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(bsz, -1, -1)
    tokens = torch.cat([cls_token, tokens], dim=1)
    tokens = tokens + clip_model.visual.positional_embedding[: tokens.size(1)].unsqueeze(0)

    tokens = clip_model.visual.ln_pre(tokens)
    tokens = clip_model.visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)

    orig_cls = clip_model.visual.ln_post(tokens[:, 0, :])
    if clip_model.visual.proj is not None:
        orig_cls = orig_cls @ clip_model.visual.proj
    orig_cls = F.normalize(orig_cls, dim=-1)

    patch_tokens = tokens[:, 1:, :]
    return patch_tokens, orig_cls


def cls_from_weighted_patch_tokens(
    clip_model,
    patch_tokens: torch.Tensor,
    keep_weights: torch.Tensor,
) -> torch.Tensor:
    weighted_tokens = patch_tokens * keep_weights.unsqueeze(-1)

    cls = clip_model.visual.class_embedding.to(weighted_tokens.dtype).unsqueeze(0)
    cls = cls.expand(weighted_tokens.size(0), -1, -1)

    sequence = torch.cat([cls, weighted_tokens], dim=1)
    pos = clip_model.visual.positional_embedding[: sequence.size(1)].unsqueeze(0).to(sequence.dtype)
    sequence = clip_model.visual.ln_pre(sequence + pos)

    z = clip_model.visual.transformer(sequence.permute(1, 0, 2)).permute(1, 0, 2)
    cls_token = clip_model.visual.ln_post(z[:, 0])
    if clip_model.visual.proj is not None:
        cls_token = cls_token @ clip_model.visual.proj
    return F.normalize(cls_token, dim=-1)


def differentiable_fitness_loss(
    clip_model,
    patch_tokens: torch.Tensor,
    keep_probs: torch.Tensor,
    text_features: torch.Tensor,
    orig_cls: torch.Tensor,
    keep_penalty_value: float,
    confidence_weight: float,
    feature_weight: float,
) -> tuple[torch.Tensor, dict]:
    masked_cls = cls_from_weighted_patch_tokens(clip_model, patch_tokens, keep_probs)

    conf = F.cosine_similarity(masked_cls.unsqueeze(1), text_features.unsqueeze(0), dim=-1).max(dim=-1).values
    feat = F.cosine_similarity(orig_cls, masked_cls, dim=-1)
    base = confidence_weight * conf + feature_weight * feat

    keep_ratio = keep_probs.mean(dim=-1)
    penalty = keep_penalty_value * keep_ratio
    fitness_score = torch.clamp(base - penalty, min=0.0, max=1.0)
    fitness_loss = (1.0 - fitness_score).mean()

    stats = {
        "confidence": conf.mean().item(),
        "feature_preservation": feat.mean().item(),
        "keep_ratio": keep_ratio.mean().item(),
        "penalty": penalty.mean().item(),
        "fitness_score": fitness_score.mean().item(),
    }
    return fitness_loss, stats


def dice_set_loss(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def count_match_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    normalize_by_tokens: bool = True,
) -> torch.Tensor:
    pred_count = probs.sum(dim=1)
    tgt_count = targets.sum(dim=1)
    abs_diff = (pred_count - tgt_count).abs()
    if normalize_by_tokens:
        abs_diff = abs_diff / float(max(1, probs.size(1)))
    return abs_diff.mean()


def adaptive_topk_mask_from_probs(
    probs: torch.Tensor,
    min_keep_tokens: int,
    max_keep_tokens: int,
) -> torch.Tensor:
    bsz, n_tokens = probs.shape
    min_keep = max(1, min(int(min_keep_tokens), n_tokens))
    if max_keep_tokens <= 0:
        max_keep = n_tokens
    else:
        max_keep = max(min_keep, min(int(max_keep_tokens), n_tokens))

    soft_keep = torch.round(probs.sum(dim=1)).to(torch.int64)
    soft_keep = torch.clamp(soft_keep, min=min_keep, max=max_keep)

    mask = torch.zeros_like(probs, dtype=torch.bool)
    for i in range(bsz):
        k = int(soft_keep[i].item())
        idx = torch.topk(probs[i], k=k, dim=-1).indices
        mask[i, idx] = True
    return mask


def hard_set_accuracy_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    min_keep_tokens: int,
    max_keep_tokens: int,
) -> float:
    with torch.no_grad():
        pred_mask = adaptive_topk_mask_from_probs(probs, min_keep_tokens, max_keep_tokens).float()
        return (pred_mask == targets).float().mean().item()


def keep_ratio_from_probs(
    probs: torch.Tensor,
    min_keep_tokens: int,
    max_keep_tokens: int,
) -> float:
    with torch.no_grad():
        pred_mask = adaptive_topk_mask_from_probs(probs, min_keep_tokens, max_keep_tokens)
        return pred_mask.float().mean().item()


def build_selector(mlp: str, device: str, dropout_rate: float, mixer_mlp_ratio: [0.5,4.0]):
    outputs_are_probs = False
    criterion: nn.Module = nn.BCEWithLogitsLoss()

    if mlp == "PatchSelector":
        selector = PatchSelector(mixer_mlp_ratio=mixer_mlp_ratio).to(device)
    elif mlp == "PatchSelectorWithSoftmax":
        selector = PatchSelectorWithSoftmax().to(device)
        criterion = nn.BCELoss()
        outputs_are_probs = True
    elif mlp == "SimplePatchSelector":
        selector = SimplePatchSelector().to(device)
    elif mlp == "SimplePatchSelectorWithDropout":
        selector = SimplePatchSelectorWithDropout(dropout=dropout_rate).to(device)
    elif mlp == "PatchSelectorResBlock":
        selector = PatchSelectorResBlock().to(device)
    elif mlp == "LightweightPatchSelector":
        selector = LightweightPatchSelector(dropout=dropout_rate).to(device)
    else:
        raise ValueError(f"Unknown MLP type: {mlp}")

    return selector, criterion, outputs_are_probs


def selector_outputs_to_probs_and_cls_loss(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    outputs_are_probs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if outputs_are_probs:
        probs = logits_or_probs.clamp(1e-6, 1 - 1e-6)
        cls_loss = criterion(probs, targets)
    else:
        probs = torch.sigmoid(logits_or_probs)
        cls_loss = criterion(logits_or_probs, targets)

    if probs.shape != targets.shape:
        raise ValueError(f"Selector output shape {probs.shape} does not match targets shape {targets.shape}")

    return probs, cls_loss


def build_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: dict, total_epochs: int):
    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
    if scheduler_name in {"none", "off", ""}:
        return None, False
    if scheduler_name == "step":
        step_size = int(scheduler_cfg.get("step_size", max(1, total_epochs // 3)))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), False
    if scheduler_name == "cosine":
        t_max = int(scheduler_cfg.get("t_max", total_epochs))
        eta_min = float(scheduler_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min), False
    if scheduler_name == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 3))
        eta_min = float(scheduler_cfg.get("min_lr", 1e-6))
        mode = str(scheduler_cfg.get("mode", "min"))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=eta_min,
        ), True
    raise ValueError(f"Unsupported lr_scheduler name: {scheduler_name}")


def resolve_resume_checkpoint(save_dir: str, requested_path: str) -> Optional[str]:
    if requested_path:
        path = requested_path.strip()
        if path:
            if os.path.isabs(path):
                return path
            return os.path.join(save_dir, path)

    candidates = sorted(Path(save_dir).glob("checkpoint_epoch_*.pt"))
    if not candidates:
        return None
    return str(candidates[-1])


def load_metric_from_checkpoint(path: str, key: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    value = checkpoint.get(key)
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GA-aligned patch selector with set + count + fitness losses.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "training_config_ga_distill.yaml"))
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_config_placeholders(cfg)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples = int(cfg["num_samples"])
    dataset_name = cfg["dataset_name"]
    data_dir = cfg["data_dir"]
    use_local_data = bool(cfg["use_local_data"])
    data_split = cfg["split"]
    cache_dir = cfg["cache_dir"]
    model_id = cfg["model_id"]
    annotation_path = cfg["annotation_path"]
    img_size = int(cfg["img_size"])
    batch_size = int(cfg["batch_size"])
    seed = int(cfg["seed"])
    num_workers = int(cfg["num_workers"])
    epochs = int(cfg["epochs"])
    save_dir = str(cfg["save_dir"])
    exp_name = str(cfg["exp_name"])
    mlp = str(cfg["mlp"])
    mixer_mlp_ratio = cfg.get("mixer_mlp_ratio", [0.5, 4.0])
    keep_penalty = float(cfg.get("keep_penalty", 0.0))
    resume_training = bool(cfg.get("resume_training", False))
    resume_checkpoint_path = str(cfg.get("resume_checkpoint_path", ""))
    resume_load_optimizer = bool(cfg.get("resume_load_optimizer", True))

    selection_cfg = cfg.get("selection", {})
    min_keep_tokens = int(selection_cfg.get("min_keep_tokens", cfg.get("min_keep_tokens", 1)))
    max_keep_tokens = int(selection_cfg.get("max_keep_tokens", cfg.get("max_keep_tokens", 0)))

    weight_decay = float(cfg.get("weight_decay", 0.01))
    selector_dropout = float(cfg.get("selector_dropout", 0.1))
    lr_scheduler_cfg = cfg.get("lr_scheduler", {"name": "none"})
    if isinstance(lr_scheduler_cfg, str):
        lr_scheduler_cfg = {"name": lr_scheduler_cfg}

    weights_cfg = cfg.get("weights", {})
    bce_loss_weight = float(weights_cfg.get("bce_loss", 1.0))
    dice_loss_weight = float(weights_cfg.get("dice_loss", 1.0))
    count_loss_weight = float(weights_cfg.get("count_loss", 1.0))
    fitness_loss_weight = float(weights_cfg.get("fitness_loss", 0.0))
    fitness_conf_weight = float(weights_cfg.get("conf", 0.4))
    fitness_feat_weight = float(weights_cfg.get("feat", 0.6))
    fitness_log_interval = int(cfg.get("fitness_log_interval", 50))

    save_dir = f"{save_dir}/{exp_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    training_log_path = os.path.join(save_dir, "training_log.txt")
    log_mode = "a" if resume_training and os.path.exists(training_log_path) else "w"
    with open(training_log_path, log_mode, encoding="utf-8"):
        pass

    cfg_out_path = os.path.join(save_dir, "training_config.yaml")
    shutil.copy(args.config, cfg_out_path)

    log_message(f"Loading config from {args.config}", training_log_path)
    log_message(f"save_dir={save_dir}", training_log_path)
    log_message(
        f"Loss weights: bce={bce_loss_weight}, dice={dice_loss_weight}, count={count_loss_weight}, fit={fitness_loss_weight}",
        training_log_path,
    )
    log_message(
        f"Selection mode: adaptive_topk, min_keep_tokens={min_keep_tokens}, max_keep_tokens={max_keep_tokens}",
        training_log_path,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if use_local_data:
        ds, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        ds, prompts = load_data_normal(dataset_name, num_samples, data_split)

    full_dataset = PatchIndexDataset(ds=ds, jsonl_path=annotation_path, img_size=img_size)
    train_subset, val_subset, _ = split_dataset(full_dataset, ratios=(0.7, 0.15, 0.15), seed=seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    clip_model, _ = clip.load(model_id, device)
    clip_model = clip_model.float().eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    clip_input_resolution = int(getattr(clip_model.visual, "input_resolution", img_size))
    clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    selector, criterion, outputs_are_probs = build_selector(mlp, device, selector_dropout, mixer_mlp_ratio)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(cfg["lr"]), weight_decay=weight_decay)
    scheduler, scheduler_on_val = build_lr_scheduler(optimizer, lr_scheduler_cfg, epochs)

    start_epoch = 1
    best_specs = {
        "total": ("val_total_loss", "checkpoint_best_total.pt"),
        "bce": ("val_bce_loss", "checkpoint_best_bce.pt"),
        "dice": ("val_dice_loss", "checkpoint_best_dice.pt"),
        "count": ("val_count_loss", "checkpoint_best_count.pt"),
        "fitness": ("val_fitness_loss", "checkpoint_best_fitness.pt"),
    }
    best_metrics = {name: None for name in best_specs}
    for name, (metric_key, filename) in best_specs.items():
        best_metrics[name] = load_metric_from_checkpoint(os.path.join(save_dir, filename), metric_key)

    resumed_ckpt = None

    if resume_training:
        resume_path = resolve_resume_checkpoint(save_dir, resume_checkpoint_path)
        if resume_path is None or not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume requested but checkpoint not found: {resume_path}")

        resumed_ckpt = torch.load(resume_path, map_location=device)
        selector.load_state_dict(resumed_ckpt["model_state_dict"])
        if resume_load_optimizer and "optimizer_state_dict" in resumed_ckpt:
            optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in resumed_ckpt:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
        start_epoch = int(resumed_ckpt.get("epoch", 0)) + 1

        resume_best_keys = {
            "total": "best_val_total_loss",
            "bce": "best_val_bce_loss",
            "dice": "best_val_dice_loss",
            "count": "best_val_count_loss",
            "fitness": "best_val_fitness_loss",
        }
        for name, key in resume_best_keys.items():
            if best_metrics[name] is None and resumed_ckpt.get(key) is not None:
                best_metrics[name] = float(resumed_ckpt[key])
        log_message(f"Resumed from {resume_path} at epoch {start_epoch}", training_log_path)

    log_message(
        (
            "Initial best losses: "
            f"total={best_metrics['total']}, "
            f"bce={best_metrics['bce']}, "
            f"dice={best_metrics['dice']}, "
            f"count={best_metrics['count']}, "
            f"fitness={best_metrics['fitness']}"
        ),
        training_log_path,
    )

    for epoch in range(start_epoch, epochs + 1):
        selector.train()
        train_total = 0.0
        train_bce = 0.0
        train_dice = 0.0
        train_count = 0.0
        train_fit = 0.0
        train_acc = 0.0
        train_keep = 0.0
        steps = 0

        for imgs, targets in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            patch_tokens, orig_cls = extract_patch_and_cls_tokens_batch(
                clip_model,
                imgs,
                clip_input_resolution,
                clip_mean,
                clip_std,
            )

            raw = selector(patch_tokens).squeeze(-1)
            probs, bce_loss = selector_outputs_to_probs_and_cls_loss(raw, targets, criterion, outputs_are_probs)
            dice_loss = dice_set_loss(probs, targets)
            cnt_loss = count_match_loss(probs, targets, normalize_by_tokens=True)

            if fitness_loss_weight > 0.0:
                fit_loss, fit_stats = differentiable_fitness_loss(
                    clip_model=clip_model,
                    patch_tokens=patch_tokens,
                    keep_probs=probs,
                    text_features=text_features,
                    orig_cls=orig_cls,
                    keep_penalty_value=keep_penalty,
                    confidence_weight=fitness_conf_weight,
                    feature_weight=fitness_feat_weight,
                )
            else:
                fit_loss = torch.zeros((), device=device)
                fit_stats = {
                    "fitness_score": 0.0,
                    "confidence": 0.0,
                    "feature_preservation": 0.0,
                    "keep_ratio": 0.0,
                }

            total_loss = (
                bce_loss_weight * bce_loss
                + dice_loss_weight * dice_loss
                + count_loss_weight * cnt_loss
                + fitness_loss_weight * fit_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            train_total += total_loss.item()
            train_bce += bce_loss.item()
            train_dice += dice_loss.item()
            train_count += cnt_loss.item()
            train_fit += fit_loss.item()
            train_acc += hard_set_accuracy_from_probs(probs, targets, min_keep_tokens, max_keep_tokens)
            train_keep += keep_ratio_from_probs(probs, min_keep_tokens, max_keep_tokens)
            steps += 1

            if fitness_log_interval > 0 and steps % fitness_log_interval == 0:
                log_message(
                    (
                        f"Epoch {epoch}/{epochs} Step {steps}: total={total_loss.item():.4f}, "
                        f"bce={bce_loss.item():.4f}, dice={dice_loss.item():.4f}, count={cnt_loss.item():.4f}, "
                        f"fit={fit_loss.item():.4f}, fit_score={fit_stats['fitness_score']:.4f}, "
                        f"conf={fit_stats['confidence']:.4f}, feat={fit_stats['feature_preservation']:.4f}"
                    ),
                    training_log_path,
                )

        avg_train_total = train_total / max(1, steps)
        avg_train_bce = train_bce / max(1, steps)
        avg_train_dice = train_dice / max(1, steps)
        avg_train_count = train_count / max(1, steps)
        avg_train_fit = train_fit / max(1, steps)
        avg_train_acc = train_acc / max(1, steps)
        avg_train_keep = train_keep / max(1, steps)

        selector.eval()
        val_total = 0.0
        val_bce = 0.0
        val_dice = 0.0
        val_count = 0.0
        val_fit = 0.0
        val_acc = 0.0
        val_keep = 0.0
        val_steps = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                patch_tokens, orig_cls = extract_patch_and_cls_tokens_batch(
                    clip_model,
                    imgs,
                    clip_input_resolution,
                    clip_mean,
                    clip_std,
                )

                raw = selector(patch_tokens).squeeze(-1)
                probs, bce_loss = selector_outputs_to_probs_and_cls_loss(raw, targets, criterion, outputs_are_probs)
                dice_loss = dice_set_loss(probs, targets)
                cnt_loss = count_match_loss(probs, targets, normalize_by_tokens=True)

                if fitness_loss_weight > 0.0:
                    fit_loss, _ = differentiable_fitness_loss(
                        clip_model=clip_model,
                        patch_tokens=patch_tokens,
                        keep_probs=probs,
                        text_features=text_features,
                        orig_cls=orig_cls,
                        keep_penalty_value=keep_penalty,
                        confidence_weight=fitness_conf_weight,
                        feature_weight=fitness_feat_weight,
                    )
                else:
                    fit_loss = torch.zeros((), device=device)

                total_loss = (
                    bce_loss_weight * bce_loss
                    + dice_loss_weight * dice_loss
                    + count_loss_weight * cnt_loss
                    + fitness_loss_weight * fit_loss
                )

                val_total += total_loss.item()
                val_bce += bce_loss.item()
                val_dice += dice_loss.item()
                val_count += cnt_loss.item()
                val_fit += fit_loss.item()
                val_acc += hard_set_accuracy_from_probs(probs, targets, min_keep_tokens, max_keep_tokens)
                val_keep += keep_ratio_from_probs(probs, min_keep_tokens, max_keep_tokens)
                val_steps += 1

        avg_val_total = val_total / max(1, val_steps)
        avg_val_bce = val_bce / max(1, val_steps)
        avg_val_dice = val_dice / max(1, val_steps)
        avg_val_count = val_count / max(1, val_steps)
        avg_val_fit = val_fit / max(1, val_steps)
        avg_val_acc = val_acc / max(1, val_steps)
        avg_val_keep = val_keep / max(1, val_steps)

        log_message(
            (
                f"Epoch {epoch}/{epochs} Train Total: {avg_train_total:.4f} "
                f"(BCE: {avg_train_bce:.4f}, Dice: {avg_train_dice:.4f}, Count: {avg_train_count:.4f}, Fit: {avg_train_fit:.4f}) "
                f"Set Acc: {avg_train_acc:.4f} Keep%: {avg_train_keep * 100:.2f} LR: {optimizer.param_groups[0]['lr']:.6g}"
            ),
            training_log_path,
        )
        log_message(
            (
                f"Epoch {epoch}/{epochs} Val   Total: {avg_val_total:.4f} "
                f"(BCE: {avg_val_bce:.4f}, Dice: {avg_val_dice:.4f}, Count: {avg_val_count:.4f}, Fit: {avg_val_fit:.4f}) "
                f"Set Acc: {avg_val_acc:.4f} Keep%: {avg_val_keep * 100:.2f}"
            ),
            training_log_path,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": selector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_total_loss": avg_val_total,
            "val_bce_loss": avg_val_bce,
            "val_dice_loss": avg_val_dice,
            "val_count_loss": avg_val_count,
            "val_fitness_loss": avg_val_fit,
            "val_set_acc": avg_val_acc,
            "val_keep_pct": avg_val_keep * 100.0,
            "selection_mode": "adaptive_topk",
            "min_keep_tokens": min_keep_tokens,
            "max_keep_tokens": max_keep_tokens,
        }

        checkpoint["best_val_total_loss"] = best_metrics["total"]
        checkpoint["best_val_bce_loss"] = best_metrics["bce"]
        checkpoint["best_val_dice_loss"] = best_metrics["dice"]
        checkpoint["best_val_count_loss"] = best_metrics["count"]
        checkpoint["best_val_fitness_loss"] = best_metrics["fitness"]

        epoch_ckpt = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, epoch_ckpt)

        current_metrics = {
            "total": avg_val_total,
            "bce": avg_val_bce,
            "dice": avg_val_dice,
            "count": avg_val_count,
            "fitness": avg_val_fit,
        }

        for name, (metric_key, filename) in best_specs.items():
            current_value = float(current_metrics[name])
            previous_best = best_metrics[name]
            if previous_best is None or current_value < float(previous_best):
                best_metrics[name] = current_value
                checkpoint["best_val_total_loss"] = best_metrics["total"]
                checkpoint["best_val_bce_loss"] = best_metrics["bce"]
                checkpoint["best_val_dice_loss"] = best_metrics["dice"]
                checkpoint["best_val_count_loss"] = best_metrics["count"]
                checkpoint["best_val_fitness_loss"] = best_metrics["fitness"]
                best_ckpt = os.path.join(save_dir, filename)
                torch.save(checkpoint, best_ckpt)
                log_message(f"Saved new best ({metric_key}) model to {best_ckpt}", training_log_path)

        if scheduler is not None:
            if scheduler_on_val:
                scheduler.step(avg_val_total)
            else:
                scheduler.step()


if __name__ == "__main__":
    main()
