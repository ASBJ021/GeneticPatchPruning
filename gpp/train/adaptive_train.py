from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import shutil
import sys

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is available on sys.path when running the script directly
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


def log_message(message: str, log_path: Optional[str] = None) -> None:
    timestamped_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped_message)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(timestamped_message + "\n")


cfg_path = "/var/lit2425/jenga/GeneticPatchPruning/config/training_config.yaml"
log_message(f"Loading config from {cfg_path}")

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("device", "cuda")
if not torch.cuda.is_available():
    device = "cpu"

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

num_samples = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_dir = cfg["data_dir"]
use_local_data = cfg["use_local_data"]
data_split = cfg["split"]
cache_dir = cfg["cache_dir"]
model_id = cfg["model_id"]
annotation_path = cfg["annotation_path"]
img_size = cfg["img_size"]
batch_size = cfg["batch_size"]
seed = cfg["seed"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
save_dir = cfg["save_dir"]
exp_name = cfg["exp_name"]
mlp = cfg["mlp"]
keep_penalty = cfg["keep_penalty"]
resume_training = bool(cfg.get("resume_training", False))
resume_checkpoint_path = cfg.get("resume_checkpoint_path")
resume_load_optimizer = bool(cfg.get("resume_load_optimizer", True))
selection_threshold = float(cfg.get("selection_threshold", 0.5))
min_keep_tokens = int(cfg.get("min_keep_tokens", 1))
weight_decay = float(cfg.get("weight_decay", 0.01))
selector_dropout = float(cfg.get("selector_dropout", 0.1))
lr_scheduler_cfg = cfg.get("lr_scheduler", {"name": "none"})
if isinstance(lr_scheduler_cfg, str):
    lr_scheduler_cfg = {"name": lr_scheduler_cfg}

weights_cfg = cfg.get("weights", {})
fitness_weight = float(weights_cfg.get("fitness_loss", 1.0))
bce_loss_weight = float(weights_cfg.get("bce_loss", 1.0))
fitness_conf_weight = float(weights_cfg.get("conf", 0.4))
fitness_feat_weight = float(weights_cfg.get("feat", 0.6))
fitness_log_interval = int(cfg.get("fitness_log_interval", 50))
log_val_fitness_loss = bool(cfg.get("log_val_fitness_loss", True))

save_dir = f"{save_dir}/{exp_name}"
Path(save_dir).mkdir(parents=True, exist_ok=True)
training_log_path = os.path.join(save_dir, "training_log.txt")
log_mode = "a" if resume_training and os.path.exists(training_log_path) else "w"
with open(training_log_path, log_mode, encoding="utf-8"):
    pass
if log_mode == "a":
    log_message("-" * 80, training_log_path)
    log_message("Resuming training run (appending logs)", training_log_path)

log_message(f"{save_dir = }", training_log_path)
log_message(f"Training logs will be saved to: {training_log_path}", training_log_path)

config_save_path = os.path.join(save_dir, "training_config.yaml")
shutil.copy(cfg_path, config_save_path)


def accuracy_at_threshold_with_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
) -> float:
    with torch.no_grad():
        preds = (probs >= thresh).float()
        correct = (preds == targets).float().mean().item()
    return correct


def hard_selection_mask_from_probs(
    probs: torch.Tensor,
    thresh: float,
    min_keep_tokens_value: int,
) -> torch.Tensor:
    """Convert patch probabilities to a hard selection mask with a minimum keep floor."""
    hard = probs >= thresh
    if min_keep_tokens_value <= 0:
        return hard

    n_tokens = int(probs.size(1))
    min_keep = max(1, min(min_keep_tokens_value, n_tokens))
    kept = hard.sum(dim=1)
    need_fix = kept < min_keep
    if need_fix.any():
        fix_idx = torch.topk(probs[need_fix], k=min_keep, dim=1).indices
        fixed = torch.zeros_like(hard[need_fix], dtype=torch.bool)
        fixed.scatter_(1, fix_idx, True)
        hard[need_fix] = fixed
    return hard


def init_keep_tracker() -> dict:
    return {"sum": 0.0, "sum_sq": 0.0, "count": 0, "min": 1.0, "max": 0.0}


def update_keep_tracker(tracker: dict, hard_mask: torch.Tensor) -> None:
    keep_ratios = hard_mask.float().mean(dim=1)
    tracker["sum"] += keep_ratios.sum().item()
    tracker["sum_sq"] += (keep_ratios * keep_ratios).sum().item()
    tracker["count"] += int(keep_ratios.numel())
    tracker["min"] = min(tracker["min"], float(keep_ratios.min().item()))
    tracker["max"] = max(tracker["max"], float(keep_ratios.max().item()))


def finalize_keep_tracker(tracker: dict) -> tuple[float, float, float, float]:
    c = max(1, int(tracker["count"]))
    mean = tracker["sum"] / c
    mean_sq = tracker["sum_sq"] / c
    var = max(0.0, mean_sq - mean * mean)
    std = var ** 0.5
    return mean, std, float(tracker["min"]), float(tracker["max"])


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
    batch_size_local, dim, _, _ = x.shape
    tokens = x.reshape(batch_size_local, dim, -1).permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.unsqueeze(0).expand(batch_size_local, -1, -1)
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
    confidence_weight: float = 0.4,
    feature_weight: float = 0.6,
) -> tuple[torch.Tensor, dict]:
    masked_cls = cls_from_weighted_patch_tokens(clip_model, patch_tokens, keep_probs)

    conf = F.cosine_similarity(
        masked_cls.unsqueeze(1),
        text_features.unsqueeze(0),
        dim=-1,
    ).max(dim=-1).values
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


def build_selector(
    selector_name: str,
    selector_device: str,
    dropout_rate: float = 0.1,
) -> tuple[nn.Module, nn.Module, bool]:
    outputs_are_probs = False
    criterion: nn.Module = nn.BCEWithLogitsLoss()

    if selector_name == "PatchSelector":
        selector = PatchSelector().to(selector_device)
    elif selector_name == "PatchSelectorWithSoftmax":
        selector = PatchSelectorWithSoftmax().to(selector_device)
        criterion = nn.BCELoss()
        outputs_are_probs = True
    elif selector_name == "SimplePatchSelector":
        selector = SimplePatchSelector().to(selector_device)
    elif selector_name == "SimplePatchSelectorWithDropout":
        selector = SimplePatchSelectorWithDropout(dropout=dropout_rate).to(selector_device)
    elif selector_name == "PatchSelectorResBlock":
        selector = PatchSelectorResBlock().to(selector_device)
    elif selector_name == "LightweightPatchSelector":
        selector = LightweightPatchSelector(dropout=dropout_rate).to(selector_device)
    else:
        raise ValueError(f"Unknown MLP type: {selector_name}")

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


def find_latest_epoch_checkpoint(checkpoint_dir: str) -> Optional[str]:
    candidates = sorted(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    if not candidates:
        return None
    return str(candidates[-1])


def resolve_resume_checkpoint(save_directory: str, requested_path: Optional[str]) -> Optional[str]:
    if requested_path:
        path = requested_path.strip()
        if path:
            if os.path.isabs(path):
                return path
            return os.path.join(save_directory, path)
    return find_latest_epoch_checkpoint(save_directory)


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


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, target_device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict,
    total_epochs: int,
) -> tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
    if scheduler_name in {"none", "off", ""}:
        return None, False
    if scheduler_name == "step":
        step_size = int(scheduler_cfg.get("step_size", max(1, total_epochs // 3)))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), False
    if scheduler_name == "cosine":
        t_max = int(scheduler_cfg.get("t_max", total_epochs))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr), False
    if scheduler_name == "plateau":
        mode = str(scheduler_cfg.get("mode", "min"))
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 3))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        return scheduler, True
    raise ValueError(f"Unsupported lr_scheduler name: {scheduler_name}")


def main() -> None:
    log_message("Starting training run", training_log_path)
    log_message(f"{dataset_name = }", training_log_path)
    log_message(f"{num_samples = }", training_log_path)

    if use_local_data:
        log_message(f"loading data from: {data_dir}", training_log_path)
        ds, prompts = load_data_folder(data_dir, num_samples, SPLIT=data_split, cache_dir=cache_dir)
    else:
        ds, prompts = load_data_normal(dataset_name, num_samples, data_split)

    clip_model, _ = clip.load(model_id, device)
    clip_model.float()
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    clip_input_resolution = int(getattr(clip_model.visual, "input_resolution", img_size))
    clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    full_dataset = PatchIndexDataset(ds=ds, jsonl_path=annotation_path, img_size=img_size)
    num_classes = full_dataset.num_classes

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

    patch_selector, criterion, outputs_are_probs = build_selector(mlp, device, selector_dropout)
    log_message(f"Training started using MLP: {mlp}", training_log_path)

    optimizer = torch.optim.AdamW(
        patch_selector.parameters(),
        lr=cfg["lr"],
        weight_decay=weight_decay,
    )
    scheduler, scheduler_on_val = build_lr_scheduler(optimizer, lr_scheduler_cfg, epochs)
    if scheduler is None:
        log_message("LR scheduler: none", training_log_path)
    else:
        log_message(f"LR scheduler: {scheduler.__class__.__name__}", training_log_path)
    start_epoch = 1
    resumed_from: Optional[str] = None
    resumed_checkpoint: Optional[dict] = None

    if resume_training:
        resolved_resume_path = resolve_resume_checkpoint(save_dir, resume_checkpoint_path)
        if resolved_resume_path is None or not os.path.exists(resolved_resume_path):
            raise FileNotFoundError(
                f"Resume requested but checkpoint was not found. "
                f"Requested: {resume_checkpoint_path}, search_dir: {save_dir}"
            )
        resumed_from = resolved_resume_path
        resumed_checkpoint = torch.load(resolved_resume_path, map_location=device)
        patch_selector.load_state_dict(resumed_checkpoint["model_state_dict"])

        if resume_load_optimizer and "optimizer_state_dict" in resumed_checkpoint:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            log_message("Loaded optimizer state from resume checkpoint", training_log_path)
            if scheduler is not None and "scheduler_state_dict" in resumed_checkpoint:
                scheduler.load_state_dict(resumed_checkpoint["scheduler_state_dict"])
                log_message("Loaded scheduler state from resume checkpoint", training_log_path)
        else:
            log_message("Resume without optimizer state (optimizer reinitialized)", training_log_path)

        last_epoch = int(resumed_checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        log_message(
            f"Resumed model from {resumed_from} (last_epoch={last_epoch}, start_epoch={start_epoch})",
            training_log_path,
        )

    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "cfg_path": cfg_path,
        "save_dir": save_dir,
        "exp_name": exp_name,
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "split": data_split,
        "seed": seed,
        "model_class": patch_selector.__class__.__name__,
        "model_module": patch_selector.__class__.__module__,
        "num_model_params": int(sum(p.numel() for p in patch_selector.parameters())),
        "num_classes": int(num_classes),
        "loss_function": criterion.__class__.__name__,
        "loss_params": {
            "pos_weight": (
                criterion.pos_weight.detach().cpu().tolist()
                if getattr(criterion, "pos_weight", None) is not None
                else None
            )
        },
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": dict(optimizer.defaults),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "device": device,
        "clip_model_id": model_id,
        "fitness_weight": fitness_weight,
        "bce_loss_weight": bce_loss_weight,
        "fitness_confidence_weight": fitness_conf_weight,
        "fitness_feature_weight": fitness_feat_weight,
        "keep_penalty": keep_penalty,
        "selection_threshold": selection_threshold,
        "min_keep_tokens": min_keep_tokens,
        "weight_decay": weight_decay,
        "selector_dropout": selector_dropout,
        "lr_scheduler": lr_scheduler_cfg,
        "log_val_fitness_loss": log_val_fitness_loss,
        "resume_training": resume_training,
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_load_optimizer": resume_load_optimizer,
        "resumed_from": resumed_from,
        "start_epoch": int(start_epoch),
    }

    metadata_path = os.path.join(save_dir, "run_metadata.yaml")
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_metadata, f, sort_keys=False)
    log_message(f"Saved run metadata to {metadata_path}", training_log_path)

    best_val_cls_loss = load_metric_from_checkpoint(
        os.path.join(save_dir, "checkpoint_best_cls.pt"),
        "val_cls_loss",
    )
    best_val_total_loss = load_metric_from_checkpoint(
        os.path.join(save_dir, "checkpoint_best_total.pt"),
        "val_total_loss",
    )
    if best_val_cls_loss is None and resumed_checkpoint is not None:
        if resumed_checkpoint.get("val_cls_loss") is not None:
            best_val_cls_loss = float(resumed_checkpoint["val_cls_loss"])
    if best_val_total_loss is None and resumed_checkpoint is not None:
        if resumed_checkpoint.get("val_total_loss") is not None:
            best_val_total_loss = float(resumed_checkpoint["val_total_loss"])

    log_message(f"Initial best_val_cls_loss: {best_val_cls_loss}", training_log_path)
    log_message(f"Initial best_val_total_loss: {best_val_total_loss}", training_log_path)

    log_message(f"{train_loader = }", training_log_path)
    log_message(f"{val_loader = }", training_log_path)
    log_message(f"{num_classes = }", training_log_path)

    if start_epoch > epochs:
        log_message(
            f"No training needed: start_epoch={start_epoch} exceeds configured epochs={epochs}",
            training_log_path,
        )
        return
#### training ####
    for epoch in range(start_epoch, epochs + 1):
        patch_selector.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_fitness_loss = 0.0
        total_acc = 0.0
        train_keep_tracker = init_keep_tracker()
        steps = 0

        for imgs, targets in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            patches, orig_cls = extract_patch_and_cls_tokens_batch(
                clip_model,
                imgs,
                clip_input_resolution,
                clip_mean,
                clip_std,
            )

            logits_or_probs = patch_selector(patches).squeeze(-1)
            probs, cls_loss = selector_outputs_to_probs_and_cls_loss(
                logits_or_probs,
                targets,
                criterion,
                outputs_are_probs,
            )

            fitness_loss, fit_stats = differentiable_fitness_loss(
                clip_model,
                patches,
                probs,
                text_features,
                orig_cls,
                keep_penalty_value=keep_penalty,
                confidence_weight=fitness_conf_weight,
                feature_weight=fitness_feat_weight,
            )

            weighted_cls_loss = bce_loss_weight * cls_loss
            weighted_fitness_loss = fitness_weight * fitness_loss
            loss = weighted_cls_loss + weighted_fitness_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_fitness_loss += fitness_loss.item()
            hard_mask = hard_selection_mask_from_probs(
                probs,
                thresh=selection_threshold,
                min_keep_tokens_value=min_keep_tokens,
            )
            update_keep_tracker(train_keep_tracker, hard_mask)
            total_acc += accuracy_at_threshold_with_probs(probs, targets, thresh=selection_threshold)
            steps += 1

            if fitness_log_interval > 0 and steps % fitness_log_interval == 0:
                batch_keep_mean = hard_mask.float().mean().item()
                log_message(
                    (
                        f"Epoch {epoch}/{epochs} Step {steps}: "
                        f"cls_loss={cls_loss.item():.4f}, "
                        f"fitness_loss={fitness_loss.item():.4f}, "
                        f"weighted_cls={weighted_cls_loss.item():.4f}, "
                        f"weighted_fitness={weighted_fitness_loss.item():.4f}, "
                        f"total={loss.item():.4f}, "
                        f"fitness_score={fit_stats['fitness_score']:.4f}, "
                        f"conf={fit_stats['confidence']:.4f}, "
                        f"feat={fit_stats['feature_preservation']:.4f}, "
                        f"keep_ratio={fit_stats['keep_ratio']:.4f}, "
                        f"hard_keep_pct={batch_keep_mean * 100:.2f}%"
                    ),
                    training_log_path,
                )

        avg_loss = total_loss / max(1, steps)
        avg_cls_loss = total_cls_loss / max(1, steps)
        avg_fitness_loss = total_fitness_loss / max(1, steps)
        avg_acc = total_acc / max(1, steps)
        train_keep_mean, train_keep_std, train_keep_min, train_keep_max = finalize_keep_tracker(train_keep_tracker)

####### validation ######
        patch_selector.eval()
        val_total_loss = 0.0
        val_total_cls_loss = 0.0
        val_total_fitness_loss = 0.0
        val_total_acc = 0.0
        val_keep_tracker = init_keep_tracker()
        val_steps = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                patches, orig_cls = extract_patch_and_cls_tokens_batch(
                    clip_model,
                    imgs,
                    clip_input_resolution,
                    clip_mean,
                    clip_std,
                )

                logits_or_probs = patch_selector(patches).squeeze(-1)
                probs, cls_loss = selector_outputs_to_probs_and_cls_loss(
                    logits_or_probs,
                    targets,
                    criterion,
                    outputs_are_probs,
                )

                if log_val_fitness_loss:
                    fitness_loss, _ = differentiable_fitness_loss(
                        clip_model,
                        patches,
                        probs,
                        text_features,
                        orig_cls,
                        keep_penalty_value=keep_penalty,
                        confidence_weight=fitness_conf_weight,
                        feature_weight=fitness_feat_weight,
                    )
                else:
                    fitness_loss = torch.zeros((), device=cls_loss.device, dtype=cls_loss.dtype)

                weighted_cls_loss = bce_loss_weight * cls_loss
                weighted_fitness_loss = fitness_weight * fitness_loss
                total_val_loss = weighted_cls_loss + weighted_fitness_loss

                val_total_loss += total_val_loss.item()
                val_total_cls_loss += cls_loss.item()
                val_total_fitness_loss += fitness_loss.item()
                hard_mask = hard_selection_mask_from_probs(
                    probs,
                    thresh=selection_threshold,
                    min_keep_tokens_value=min_keep_tokens,
                )
                update_keep_tracker(val_keep_tracker, hard_mask)
                val_total_acc += accuracy_at_threshold_with_probs(probs, targets, thresh=selection_threshold)
                val_steps += 1

        avg_val_loss = val_total_loss / max(1, val_steps)
        avg_val_cls_loss = val_total_cls_loss / max(1, val_steps)
        avg_val_fitness_loss = val_total_fitness_loss / max(1, val_steps)
        avg_val_acc = val_total_acc / max(1, val_steps)
        val_keep_mean, val_keep_std, val_keep_min, val_keep_max = finalize_keep_tracker(val_keep_tracker)

        epoch_log = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total_loss": avg_loss,
            "train_cls_loss": avg_cls_loss,
            "train_fitness_loss": avg_fitness_loss,
            "train_bit_acc@0.5": avg_acc,
            "train_keep_pct_mean": train_keep_mean * 100.0,
            "train_keep_pct_std": train_keep_std * 100.0,
            "train_keep_pct_min": train_keep_min * 100.0,
            "train_keep_pct_max": train_keep_max * 100.0,
            "val_total_loss": avg_val_loss,
            "val_loss": avg_val_cls_loss,
            "val_cls_loss": avg_val_cls_loss,
            "val_fitness_loss": avg_val_fitness_loss,
            "val_bit_acc@0.5": avg_val_acc,
            "val_keep_pct_mean": val_keep_mean * 100.0,
            "val_keep_pct_std": val_keep_std * 100.0,
            "val_keep_pct_min": val_keep_min * 100.0,
            "val_keep_pct_max": val_keep_max * 100.0,
        }

        log_message(
            (
                f"Epoch {epoch}/{epochs} Train Total: {avg_loss:.4f} "
                f"(Cls: {avg_cls_loss:.4f}, Fit: {avg_fitness_loss:.4f}) "
                f"Bit Acc@0.5: {avg_acc:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6g} "
                f"Keep% mean/std/min/max: "
                f"{train_keep_mean * 100.0:.2f}/"
                f"{train_keep_std * 100.0:.2f}/"
                f"{train_keep_min * 100.0:.2f}/"
                f"{train_keep_max * 100.0:.2f}"
            ),
            training_log_path,
        )
        log_message(
            (
                f"Epoch {epoch}/{epochs} Val   Total: {avg_val_loss:.4f} "
                f"(Cls: {avg_val_cls_loss:.4f}, Fit: {avg_val_fitness_loss:.4f}) "
                f"Bit Acc@0.5: {avg_val_acc:.4f} "
                f"Keep% mean/std/min/max: "
                f"{val_keep_mean * 100.0:.2f}/"
                f"{val_keep_std * 100.0:.2f}/"
                f"{val_keep_min * 100.0:.2f}/"
                f"{val_keep_max * 100.0:.2f}"
            ),
            training_log_path,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": patch_selector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_class": optimizer.__class__.__name__,
            "optimizer_defaults": dict(optimizer.defaults),
            "loss_function": criterion.__class__.__name__,
            "train_total_loss": avg_loss,
            "train_loss": avg_loss,
            "train_cls_loss": avg_cls_loss,
            "train_fitness_loss": avg_fitness_loss,
            "train_keep_pct_mean": train_keep_mean * 100.0,
            "train_keep_pct_std": train_keep_std * 100.0,
            "train_keep_pct_min": train_keep_min * 100.0,
            "train_keep_pct_max": train_keep_max * 100.0,
            "val_total_loss": avg_val_loss,
            "val_loss": avg_val_cls_loss,
            "val_cls_loss": avg_val_cls_loss,
            "val_fitness_loss": avg_val_fitness_loss,
            "val_keep_pct_mean": val_keep_mean * 100.0,
            "val_keep_pct_std": val_keep_std * 100.0,
            "val_keep_pct_min": val_keep_min * 100.0,
            "val_keep_pct_max": val_keep_max * 100.0,
            "epoch_log": epoch_log,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, ckpt_path)

        if best_val_cls_loss is None or avg_val_cls_loss < best_val_cls_loss:
            best_val_cls_loss = avg_val_cls_loss
            best_ckpt_path_cls = os.path.join(save_dir, "checkpoint_best_cls.pt")
            torch.save(checkpoint, best_ckpt_path_cls)
            # Keep backward-compatible name pointing to best-by-cls checkpoint.
            best_ckpt_path_default = os.path.join(save_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_ckpt_path_default)
            log_message(
                f"Saved new best (val_cls_loss) model to {best_ckpt_path_cls}",
                training_log_path,
            )

        if best_val_total_loss is None or avg_val_loss < best_val_total_loss:
            best_val_total_loss = avg_val_loss
            best_ckpt_path_total = os.path.join(save_dir, "checkpoint_best_total.pt")
            torch.save(checkpoint, best_ckpt_path_total)
            log_message(
                f"Saved new best (val_total_loss) model to {best_ckpt_path_total}",
                training_log_path,
            )

        if scheduler is not None:
            if scheduler_on_val:
                scheduler.step(avg_val_cls_loss)
            else:
                scheduler.step()


if __name__ == "__main__":
    main()
