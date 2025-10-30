from typing import List, Tuple
import torch
import torch.nn.functional as F
import clip


def load_clip(model_id: str, device: str):
    """Load CLIP model and its preprocess/processor."""
    model, processor = clip.load(model_id, device)
    model = model.float()
    return model, processor


@torch.no_grad()
def prepare_inputs(model, processor, device: str, img, prompts: List[str]):
    """Prepare CLIP inputs: text features, image tensor, original CLS, patch tokens."""
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    pixel_values = processor(img).unsqueeze(0).to(device)
    x = model.visual.conv1(pixel_values)
    B, D, H, W = x.shape
    tokens = x.view(B, D, -1).permute(0, 2, 1)

    cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
    tokens = torch.cat([cls_token, tokens], dim=1)
    tokens += model.visual.positional_embedding.unsqueeze(0)

    tokens = model.visual.ln_pre(tokens).permute(1, 0, 2)
    tokens = model.visual.transformer(tokens).permute(1, 0, 2)

    cls = model.visual.ln_post(tokens[:, 0])
    if model.visual.proj is not None:
        cls = cls @ model.visual.proj
    cls /= cls.norm(dim=-1, keepdim=True)

    patch_tokens = tokens[:, 1:, :]
    return pixel_values, text_features, cls, patch_tokens


@torch.no_grad()
def mask_patches(model, device: str, pixel_values, indices_to_remove: List[int]):
    """Zero out selected patch token positions and return normalized CLS."""
    x = model.visual.conv1(pixel_values)
    B, D, H, W = x.shape
    tokens = x.view(B, D, -1).permute(0, 2, 1)

    tokens[:, indices_to_remove, :] = 0

    cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
    tokens = torch.cat([cls_token, tokens], dim=1)
    tokens += model.visual.positional_embedding.unsqueeze(0)

    tokens = model.visual.ln_pre(tokens).permute(1, 0, 2)
    tokens = model.visual.transformer(tokens).permute(1, 0, 2)

    cls = model.visual.ln_post(tokens[:, 0])
    if model.visual.proj is not None:
        cls = cls @ model.visual.proj
    cls /= cls.norm(dim=-1, keepdim=True)
    return cls


@torch.no_grad()
def forward_with_selected_patches(model, device: str, x, selected_indices: List[int]):
    """Forward pass keeping only CLS and selected patch tokens; returns normalized img feature."""
    cls = model.visual.class_embedding.unsqueeze(0).expand(1, -1, -1)
    sequence = torch.cat([cls, x], dim=1)
    pos = model.visual.positional_embedding[:sequence.size(1)].unsqueeze(0)
    sequence = model.visual.ln_pre(sequence + pos)

    keep_idx = torch.tensor([0] + [i + 1 for i in selected_indices], device=device)
    sequence = sequence[:, keep_idx, :]

    z = model.visual.transformer(sequence.permute(1, 0, 2))
    cls_token = model.visual.ln_post(z.permute(1, 0, 2)[:, 0])
    img_feat = cls_token @ model.visual.proj if model.visual.proj is not None else cls_token
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return img_feat


@torch.no_grad()
def cls_from_masked_tokens(model, device: str, x, indices_to_remove: List[int]):
    """Compute normalized CLS given pre-LN patch tokens `x` with some indices zeroed.

    Parameters
    - x: tensor of shape (B, N, D) representing patch tokens BEFORE adding CLS/pos and LN
    - indices_to_remove: patch indices to zero (mask out)
    """
    # clone to avoid side-effects across fitness calls
    tokens = x.clone()
    if len(indices_to_remove) > 0:
        tokens[:, indices_to_remove, :] = 0

    cls = model.visual.class_embedding.unsqueeze(0).expand(tokens.size(0), -1, -1)
    sequence = torch.cat([cls, tokens], dim=1)
    pos = model.visual.positional_embedding[:sequence.size(1)].unsqueeze(0)
    sequence = model.visual.ln_pre(sequence + pos)

    z = model.visual.transformer(sequence.permute(1, 0, 2))
    cls_token = model.visual.ln_post(z.permute(1, 0, 2)[:, 0])
    img_feat = cls_token @ model.visual.proj if model.visual.proj is not None else cls_token
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    return img_feat
