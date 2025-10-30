from typing import List, Tuple
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import yaml
from timm.models.mlp_mixer import MixerBlock
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from gpp.dataset.data_utils import load_data_normal, build_dataloaders, PatchIndexDataset, _load_jsonl, split_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



# cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
cfg_path = '/var/lit2425/jenga/GeneticPatchPruning/config/config.yaml'
print (f'Loading config from {cfg_path}')

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("device", "cuda")
if not torch.cuda.is_available():
    device = "cpu"


to_pil = transforms.ToPILImage()
num_samples  = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_split = cfg["split"]
model_id     = cfg["model_id"]
vis = cfg["visualize"]
annotation_path = cfg["annotation_path"]
img_size = cfg["img_size"]
batch_size = cfg["batch_size"]
seed = cfg["seed"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]

model, processor = clip.load(model_id, device)  # Load on CPU initially
model.float()  # Ensure model is in float32

class PatchSelector(nn.Module):
    def __init__(self, in_embed_dim=768, out_embed_dim=196) -> None:
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(in_embed_dim)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=in_embed_dim, nhead=8)
        #self.in_conv = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.in_conv = nn.Sequential(
            MixerBlock(in_embed_dim, 196)
        )
        self.fc = nn.Linear(in_embed_dim, 1)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, H, W)
        # x: [B, 196, 768]
        x = self.adaptive_pool(x)
        x = self.in_conv(x)
        #x = self.fc(x)
        # x = x[:, 1:,:] # [B, 768]
        x = self.fc(x) # [B, 196, 1]
        return x
    

def images_to_patches(imgs):
    patches = []
    for img in imgs:
        pil_img = to_pil(img.cpu())
        pixel_values = processor(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.visual.conv1(pixel_values)
            B, D, H, W = x.shape
            tokens = x.reshape(B, D, -1).permute(0, 2, 1)

            cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
            tokens += model.visual.positional_embedding.unsqueeze(0)

            tokens = model.visual.ln_pre(tokens)
            tokens = tokens.permute(1, 0, 2)
            tokens = model.visual.transformer(tokens)
            tokens = tokens.permute(1, 0, 2)

            orig_cls_token = tokens[:, 0, :]
            orig_cls_token = model.visual.ln_post(orig_cls_token)

            if model.visual.proj is not None:
                orig_cls_token = orig_cls_token @ model.visual.proj

            orig_cls_token /= orig_cls_token.norm(dim=-1, keepdim=True)
            patch_tokens = tokens[:, 1:, :]
            patches.append(patch_tokens)
    return patches


def main():

    # load dataset
    print(f'{dataset_name = }')
    print(f'{num_samples = }')
    ds, _ = load_data_normal(dataset_name, num_samples, data_split)
    # print(ds)

    full_dataset = PatchIndexDataset(ds=ds, jsonl_path=annotation_path, img_size=img_size)
    num_classes = full_dataset.num_classes

    train_subset, val_subset, test_subset = split_dataset(full_dataset, ratios=(0.7, 0.15, 0.15), seed=seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # print(f'{train_loader = }')

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    patch_selector = PatchSelector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])


    print(f'{train_loader = }')
    print(f'{val_loader = }')
    print(f'{num_classes = }')


    for epoch in range(1, epochs+1):
        patch_selector.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0


        for imgs, targets in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            # print(imgs.shape)
        
            targets = targets.to(device, non_blocking=True)
            # print(targets.shape)

            # convert images to clip [patches]
            patches = images_to_patches(imgs)
            patches = torch.cat(patches, dim=0).to(device)
            print(len(patches))
            print(patches[0].shape)

            logits = patch_selector(patches).squeeze(-1) 
            print(logits.shape)

            loss = criterion(logits, targets)
            print(loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # print(f'{loss = }')
            optimizer.step()

            total_loss += loss.item()
            # total_acc += accuracy_at_threshold(logits.detach(), targets)
            steps += 1
            # print(f'{total_loss = }')
            # break
        print(f'{steps = }')
        avg_loss = total_loss / max(1, steps)

        print(f'{avg_loss = }')
        # avg_acc = total_acc / max(1, steps)

        # Validation
        patch_selector.eval()
        v_total = 0.0
        # v_acc = 0.0
        v_steps = 0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", leave=False):
                # imgs = imgs.to(device, non_blocking=True)
                # targets = targets.to(device, non_blocking=True)
                imgs = imgs.to(device, non_blocking=True)
                # print(imgs.shape)
            
                targets = targets.to(device, non_blocking=True)
                # print(targets.shape)

                # convert images to clip [patches]
                patches = images_to_patches(imgs)
                patches = torch.cat(patches, dim=0).to(device)

                # logits = model(imgs)
                logits = patch_selector(patches).squeeze(-1) 
                loss = criterion(logits, targets)
                v_total += loss.item()
                # v_acc += accuracy_at_threshold(logits, targets)
                v_steps += 1
        print(f'val steps {v_steps= }')

        val_loss = v_total / max(1, v_steps)
        # val_acc = v_acc / max(1, v_steps)
        print(f'{val_loss = }')




    # cnn_model = PatchSelector(out_embed_dim=256).to(device)
    # print(f'CLIP-based Patch Selection Model: {cnn_model}')


if __name__ == "__main__":
    main()
