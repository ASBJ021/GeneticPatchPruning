from typing import List
import torch
import torch.nn.functional as F

from ..model.clip_model import mask_patches, cls_from_masked_tokens


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
    logits = masked_cls @ text_features.T
    return logits.softmax(dim=-1).max(dim=-1)[0].item()


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
    weights = weights or {"confidence": 0.4, "feature": 0.6}
    conf = get_confidence_score_from_tokens(model, device, x_tokens, indices, text_features)
    feat = get_feature_preservation_score_from_tokens(model, device, x_tokens, indices, original_cls)
    return weights["confidence"] * conf + weights["feature"] * feat
