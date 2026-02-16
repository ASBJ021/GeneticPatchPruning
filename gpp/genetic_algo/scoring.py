from typing import List
import torch
import torch.nn.functional as F

from ..model.clip_model import mask_patches, cls_from_masked_tokens
from .io import save_record_jsonl


@torch.no_grad()
def get_confidence_score(model, device: str, pixel_values, indices_to_remove: List[int], text_features) -> float:
    """Return max softmax confidence over prompts after masking."""
    masked_cls = mask_patches(model, device, pixel_values, indices_to_remove)
    logits = masked_cls @ text_features.T
    return logits.softmax(dim=-1).max(dim=-1)[0].item()


@torch.no_grad()
def get_feature_preservation_score(model, device: str, pixel_values, indices_to_remove: List[int], original_cls) -> float:
    """Return cosine similarity between original CLS and masked CLS."""
    masked_cls = mask_patches(model, device, pixel_values, indices_to_remove)
    return F.cosine_similarity(original_cls, masked_cls).item()


def fitness_function(
    model,
    device: str,
    pixel_values,
    indices: List[int],
    text_features,
    original_cls,
    weights=None,
) -> float:
    """Weighted sum of confidence and embedding preservation.

    Note: `indices` are treated as indices-to-remove for masking, mirroring
    the original behavior.
    """
    weights = weights or {"confidence": 0.4, "feature": 0.6}
    conf = get_confidence_score(model, device, pixel_values, indices, text_features)
    feat = get_feature_preservation_score(model, device, pixel_values, indices, original_cls)
    return weights["confidence"] * conf + weights["feature"] * feat


# Faster variants that reuse cached pre-LN patch tokens (x)
@torch.no_grad()
def get_confidence_score_from_tokens(model, device: str, x_tokens, indices_to_remove: List[int], text_features) -> float:
    masked_cls = cls_from_masked_tokens(model, device, x_tokens, indices_to_remove)
    # logits = masked_cls @ text_features.T
    # logits = F.layer_norm(logits, logits.shape[-1:])  # Normalize for better cosine sim
    # logits = F.cosine_similarity(masked_cls, text_features).item()
    use_cosine_similarity = True
    if use_cosine_similarity:
        cos_sim = F.cosine_similarity(
            masked_cls.unsqueeze(1),
            text_features.unsqueeze(0),
            dim=-1,
        )
        conf = cos_sim.max(dim=-1)[0]
    else:
        # masked_cls_norm = F.normalize(masked_cls)
        logits = masked_cls@ text_features.T
        logits = F.normalize(logits)
        probs = logits.softmax(dim = -1)
        conf = probs.max(dim = -1)[0]
    return conf.item()


@torch.no_grad()
def get_feature_preservation_score_from_tokens(model, device: str, x_tokens, indices_to_remove: List[int], original_cls) -> float:
    masked_cls = cls_from_masked_tokens(model, device, x_tokens, indices_to_remove)
    return F.cosine_similarity(original_cls, masked_cls).item()


def fitness_function_from_tokens(
    model,
    device: str,
    x_tokens,
    indices: List[int],
    text_features,
    original_cls,
    weights=None,

):
    save_fitness_scores = False
    fitness_log_path = "./logs/new_fitness_logs_cosine_cs/fitness_scores_cls_cosine.jsonl"
    weights = weights or {"confidence": 0.4, "feature": 0.6}
    conf = get_confidence_score_from_tokens(model, device, x_tokens, indices, text_features)
    # print(f"{conf.shape = }")
    feat = get_feature_preservation_score_from_tokens(model, device, x_tokens, indices, original_cls)
    # print(f'{feat.shape = }')
    score = weights["confidence"] * conf + weights["feature"] * feat

    if save_fitness_scores:
        save_record_jsonl(
            {
                "confidence_score": conf,
                "feature_preservation_score": feat,
                "fitness_score": score,
            },
            fitness_log_path,
        )
    # print(f"Logged fitness scores to {fitness_log_path}")

    return score, conf, feat
