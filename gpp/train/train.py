from typing import List, Tuple, Optional
import os
import sys
from pathlib import Path

# Ensure project root is available on sys.path when running the script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from gpp.model.model import PatchSelector, images_to_patches


cfg_path = '/var/lit2425/jenga/GeneticPatchPruning/config/training_config.yaml'
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
save_dir = cfg["save_dir"]
exp_name = cfg["exp_name"]

save_dir = f'{save_dir}/{exp_name}'
print(f'{save_dir = }')

Path(save_dir).mkdir(parents=True, exist_ok=True)



model, processor = clip.load(model_id, device)  # Load on CPU initially
model.float()  # Ensure model is in float32

def accuracy_at_threshold(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> float:
    """Simple multi-label accuracy: fraction of correctly predicted bits over all bits."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= thresh).float()
        correct = (preds == targets).float().mean().item()
    return correct


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
    optimizer = torch.optim.Adam(patch_selector.parameters(), lr=cfg["lr"])

    best_val_loss: Optional[float] = None
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    history_csv = os.path.join(save_dir, "history.csv")
    history_rows = []


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
            # print(len(patches))
            # print(patches[0].shape)

            logits = patch_selector(patches).squeeze(-1) 
            # print(logits.shape)

            loss = criterion(logits, targets)
            # print(loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # print(f'{loss = }')
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_at_threshold(logits.detach(), targets)
            steps += 1
            # print(f'{total_loss = }')
            # break
        # print(f'{steps = }')
        avg_loss = total_loss / max(1, steps)
        avg_acc = total_acc / max(1, steps)

        # print(f'{avg_loss = }')
        # print(f'{avg_acc = }')
        

        # Validation
        patch_selector.eval()
        val_total_loss = 0.0
        val_total_acc = 0.0
        val_steps = 0
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
                val_total_loss += loss.item()
                val_total_acc += accuracy_at_threshold(logits, targets)
                val_steps += 1
        # print(f'val steps {val_steps }')

        avg_val_loss = val_total_loss / max(1, val_steps)
        avg_val_acc = val_total_acc / max(1, val_steps)
        # val_acc = v_acc / max(1, v_steps)
        # print(f'{avg_val_loss = }')
        # print(f'{avg_val_acc = }')

        log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_bit_acc@0.5": avg_acc,
            "val_loss": avg_val_loss,
            "val_bit_acc@0.5": avg_val_acc,
        }
        print(f"Epoch {epoch}/{epochs} Train Loss: {avg_loss:.4f} Bit Acc@0.5: {avg_acc:.4f}")
        print(f"Epoch {epoch}/{epochs} Val   Loss: {avg_val_loss:.4f} Bit Acc@0.5: {avg_val_acc:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": patch_selector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
        }
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, ckpt_path)

        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved new best model to {best_ckpt_path}")





if __name__ == "__main__":
    main()
