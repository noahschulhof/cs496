import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import get_cifair10_datasets
import random
import pickle
import os

# CIFAR-10 stats
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.229, 0.224, 0.225]

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

def get_data():
    train_dataset, test_dataset = get_cifair10_datasets()
    calib_dataset, val_dataset, _ = random_split(test_dataset, [5000, 5000, 0])
    return train_dataset, calib_dataset, val_dataset

def denormalize(tensor):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)

def add_image_to_plot(tensor, ax, title):
    img = denormalize(tensor).permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def return_one_image_all_perturb(dataset, image_idx=0):
    # original image
    original_image, label = dataset[image_idx]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Original + Gaussian Noise
    add_image_to_plot(original_image, axes[0, 0], 'Original')    
    noise_levels = [0.05, 0.1, 0.2]
    for i, std in enumerate(noise_levels):
        perturbed_ds = gaussian_noise_dataset(dataset, std=std)
        perturbed_img, _ = perturbed_ds[image_idx]
        add_image_to_plot(perturbed_img, axes[0, i+1], f'Gaussian Noise\nstd={std}')
    
    # Row 2: Blur variations
    blur_params = [(3, 1.0), (5, 1.5), (7, 2.0)]
    add_image_to_plot(original_image, axes[1, 0], 'Original')
    for i, (kernel, sigma) in enumerate(blur_params):
        perturbed_ds = blur_dataset(dataset, kernel_size=kernel, sigma=sigma)
        perturbed_img, _ = perturbed_ds[image_idx]
        add_image_to_plot(perturbed_img, axes[1, i+1], f'Blur\nkernel={kernel}, σ={sigma}')
    
    # Row 3: Brightness, Erasure, JPEG
    add_image_to_plot(original_image, axes[2, 0], 'Original')
    
    # Brightness
    perturbed_ds = brightness_dataset(dataset, brightness_factor=0.5)
    perturbed_img, _ = perturbed_ds[image_idx]
    add_image_to_plot(perturbed_img, axes[2, 1], 'Brightness\nfactor=0.5 (dark)')
    
    # Random Erasure
    perturbed_ds = random_erasure_dataset(dataset, scale=(0.1, 0.3))
    perturbed_img, _ = perturbed_ds[image_idx]
    add_image_to_plot(perturbed_img, axes[2, 2], 'Random Erasure\nscale=(0.1, 0.3)')
    
    # JPEG Compression
    perturbed_ds = jpeg_compression_dataset(dataset, quality=10)
    perturbed_img, _ = perturbed_ds[image_idx]
    add_image_to_plot(perturbed_img, axes[2, 3], 'JPEG Compression\nquality=10')
    
    plt.tight_layout()
    plt.savefig('example_perturb.png', dpi=150, bbox_inches='tight')

def save_perturbed_test_batches(test_dataset, output_dir='./data'):
    perturbations = {
        'gaussian_noise': gaussian_noise_dataset(test_dataset, std=0.1),
        'blur': blur_dataset(test_dataset, kernel_size=5, sigma=1.5),
        'brightness': brightness_dataset(test_dataset, brightness_factor=0.5),
        'erasure': random_erasure_dataset(test_dataset, scale=(0.1, 0.3)),
        'jpeg': jpeg_compression_dataset(test_dataset, quality=10),
    }
    
    # convert normalized tensor back to uint8 for CIFAR format
    def tensor_to_cifar_format(tensor):
        # Denormalize
        img = denormalize(tensor)  # Now in [0, 1]
        # Convert to uint8 (0-255)
        img = (img * 255).byte().numpy()
        # CIFAR format: (3, 32, 32) -> flatten to (3072,) in R, G, B order
        # ensure R (1024), G (1024), B (1024) contiguous ordering
        return img.reshape(3, -1).flatten()
    
    for perturb_name, perturbed_dataset in perturbations.items():
        data_list = []
        labels_list = []
        
        for i in range(len(perturbed_dataset)):
            img_tensor, label = perturbed_dataset[i]
            
            # CIFAR format
            img_cifar = tensor_to_cifar_format(img_tensor)
            data_list.append(img_cifar)
            labels_list.append(label)
        
        data_array = np.vstack(data_list)
        labels_array = np.array(labels_list)
        
        batch_dict = {
            b'data': data_array,
            b'labels': labels_array,
            b'batch_label': f'test batch - {perturb_name}'.encode(),
            b'filenames': [f'{perturb_name}_{i:05d}'.encode() for i in range(len(labels_array))]
        }
        
        # save
        output_path = os.path.join(output_dir, f'test_batch_{perturb_name}')
        with open(output_path, 'wb') as f:
            pickle.dump(batch_dict, f)

if __name__ == "__main__":
    _, test_dataset = get_cifair10_datasets(root='./data', download=True)
    out_dir = './data'
    os.makedirs(out_dir, exist_ok=True)

    # save a small example visualization
    idx = random.randint(0, len(test_dataset)-1)
    return_one_image_all_perturb(test_dataset, image_idx=idx)
    print("Saved example of perturbed image: example_perturb.png")

    print("Saving perturbed test batches...")
    save_perturbed_test_batches(test_dataset, output_dir=out_dir)
    print("Done")

    # remove downloaded ciFAIR-10.zip file and batch files in ciFAIR-10 directory
    zip_path = os.path.join('./data', 'ciFAIR-10.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    ciFAIR_dir = os.path.join('./data', 'ciFAIR-10')
    if os.path.exists(ciFAIR_dir):
        for filename in os.listdir(ciFAIR_dir):
            file_path = os.path.join(ciFAIR_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(ciFAIR_dir)