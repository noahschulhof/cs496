import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import io

# CIFAR-10 stats
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]

# Precompute for clamping in normalized space: (0 - mean) / std and (1 - mean) / std
CIFAR_MIN = torch.tensor([(0 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]).view(3,1,1)
CIFAR_MAX = torch.tensor([(1 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]).view(3,1,1)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class PerturbedDataset(Dataset):
    def __init__(self, dataset, perturbation_fn):
        self.dataset = dataset
        self.perturbation_fn = perturbation_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]       # already a normalized tensor (3, 32, 32)
        return self.perturbation_fn(image), label


def _clamp_normalized(tensor):
    """Clamp tensor to valid [0,1] range accounting for CIFAR normalization."""
    return torch.max(torch.min(tensor, CIFAR_MAX), CIFAR_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GAUSSIAN NOISE
#    std  : 0.05 mild, 0.1 moderate, 0.2 heavy
#    Note : std is in normalized space, so a little goes a long way
# ─────────────────────────────────────────────────────────────────────────────
def gaussian_noise_dataset(dataset, std=0.1, mean=0.0):
    def perturb(tensor):
        noise = torch.randn_like(tensor) * std + mean
        return _clamp_normalized(tensor + noise)
    return PerturbedDataset(dataset, perturb)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN BLUR
#    kernel_size : must be odd — 3 is already strong on 32x32 images!
#    sigma       : blur radius
#    Note : images are tiny (32x32), so kernel_size=3 or 5 is plenty
# ─────────────────────────────────────────────────────────────────────────────
def blur_dataset(dataset, kernel_size=3, sigma=1.0):
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    def perturb(tensor):
        return blur(tensor)
    return PerturbedDataset(dataset, perturb)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BRIGHTNESS VARIATION
#    factor : <1 darkens, >1 brightens — applied in normalized space
#             equivalent to scaling pixel intensities before normalization
# ─────────────────────────────────────────────────────────────────────────────
def brightness_dataset(dataset, brightness_factor=0.5):
    mean = torch.tensor(CIFAR_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR_STD).view(3,1,1)

    def perturb(tensor):
        # Denormalize → adjust brightness → renormalize
        img = tensor * std + mean                           # back to [0, 1]
        img = torch.clamp(img * brightness_factor, 0, 1)   # adjust
        return (img - mean) / std                           # renormalize
    return PerturbedDataset(dataset, perturb)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RANDOM ERASURE
#    scale : (min, max) fraction of image to erase — keep small for 32x32
#    value : fill value in normalized space (0 = approx mean-grey after norm)
# ─────────────────────────────────────────────────────────────────────────────
def random_erasure_dataset(dataset, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
    erase = transforms.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=value)
    def perturb(tensor):
        return erase(tensor)
    return PerturbedDataset(dataset, perturb)


# ─────────────────────────────────────────────────────────────────────────────
# 5. JPEG COMPRESSION
#    quality : 1 (worst) → 95 (best) — must round-trip through PIL
#    Note : denormalizes → PIL JPEG encode/decode → renormalizes
# ─────────────────────────────────────────────────────────────────────────────
def jpeg_compression_dataset(dataset, quality=10):
    mean = torch.tensor(CIFAR_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR_STD).view(3,1,1)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    def perturb(tensor):
        # Denormalize to [0, 1] for PIL
        img = torch.clamp(tensor * std + mean, 0, 1)
        pil_img = to_pil(img)

        # JPEG encode/decode
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = to_tensor(Image.open(buf).convert("RGB"))  # back to [0, 1]

        # Renormalize
        return (compressed - mean) / std

    return PerturbedDataset(dataset, perturb)