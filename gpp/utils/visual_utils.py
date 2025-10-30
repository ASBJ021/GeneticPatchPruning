# visual_utils.py

import math
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt

def load_image(img_path, resize=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return image if pil else np.asarray(image).astype(np.float32) / 255.

def patchify(image_pil, resolution, patch_size, patch_stride=None):
    image_pil = image_pil.resize((resolution, resolution))
    img_t = T.ToTensor()(image_pil)  # 3×H×W
    if patch_stride is None:
        patch_stride = patch_size
    patches = img_t.unfold(1, patch_size, patch_stride) \
                  .unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size) \
                     .permute(1, 0, 2, 3)
    return patches

def viz_patches(x, figsize=(8,8), topk=None, t=5, img_title=''):
    n = x.shape[0]
    ncols = int(math.sqrt(n))
    fig, axes = plt.subplots(ncols, ncols, figsize=figsize)
    fig.suptitle(img_title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        im = (x[i].permute(1,2,0).numpy() * 255).round().astype(np.uint8)
        if topk is not None and i in set(int(j) for j in topk):
            im[:t] = im[-t:] = im[:, :t] = im[:, -t:] = (255,255,0)
        ax.imshow(im); ax.axis("off")
    plt.tight_layout(); plt.show()

def plot_heatmap_overlay(orig_img, patch_sims, grid_size, alpha=0.5, cmap="jet"):
    # build and normalize low-res heatmap
    heatmap = np.array(patch_sims).reshape(grid_size)
    # NumPy 2.0 removed ndarray.ptp(); use np.ptp(heatmap) instead
    rng = np.ptp(heatmap)
    norm = (heatmap - heatmap.min()) / (rng + 1e-8)
    # upsample and colorize
    if isinstance(orig_img, Image.Image):
        if orig_img.mode != "RGB":
            orig_img = orig_img.convert("RGB")
        base = np.array(orig_img, dtype=np.float32)
    else:
        base = np.asarray(orig_img, dtype=np.float32)
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)

    H, W = base.shape[:2]
    hm = Image.fromarray((norm * 255).astype(np.uint8), mode="L") \
           .resize((W, H), Image.BILINEAR)
    colored = plt.get_cmap(cmap)(np.array(hm) / 255.0)[..., :3]
    blended = (base * (1 - alpha) + colored * 255 * alpha).clip(0, 255).astype(np.uint8)
    plt.figure(figsize=(6,6))
    plt.imshow(blended); plt.axis("off"); plt.title("Original + Patch Heatmap")
    plt.show()

def visualize_on_original(orig_img, selected_idx):
    # draws red rectangles on a PIL image
    from clip import load as clip_load
    from torchvision.transforms import Resize, CenterCrop
    model, preprocess = clip_load()  # only to get resize/crop
    resize = next(t for t in preprocess.transforms if isinstance(t, Resize))
    crop   = next(t for t in preprocess.transforms if isinstance(t, CenterCrop))
    target = resize.size if isinstance(resize.size,int) else resize.size[0]
    crop_s = crop.size   if isinstance(crop.size,int)   else crop.size[0]
    ow,oh = orig_img.size
    nw,nh = (target, int(target*oh/ow)) if ow<oh else (int(target*ow/oh), target)
    left,top = (nw-crop_s)//2, (nh-crop_s)//2
    sx,sy = ow/nw, oh/nh
    draw = ImageDraw.Draw(orig_img)
    for ii in selected_idx:
        r,c = divmod(ii, crop_s//16)
        x0,y0 = (c*16+left)*sx, (r*16+top)*sy
        draw.rectangle([x0,y0,x0+16*sx,y0+16*sy], outline="red", width=1)
    orig_img.save("highlighted.png"); orig_img.show()
