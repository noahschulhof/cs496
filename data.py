""" ciFAIR-10 data loaders for PyTorch.

Version: 1.0

https://cvjena.github.io/cifair/
"""

import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data as data

class ciFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]

def get_cifair10_datasets(root='./data', download=True):
    # Standard CIFAR-10 normalization values (https://github.com/kuangliu/pytorch-cifar/issues/19)
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([transforms.ToTensor(), normalize,])
    
    train_dataset = ciFAIR10(root=root, train=True, download=download, transform=transform)
    test_dataset = ciFAIR10(root=root, train=False, download=download, transform=transform)
    
    return train_dataset, test_dataset


def get_cifair10_loaders(root='./data', batch_size=128, num_workers=4, download=True):
    train_dataset, test_dataset = get_cifair10_datasets(root=root, download=download)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Loading ciFAIR-10 datasets...")
    train_dataset, test_dataset = get_cifair10_datasets(download=True)
    
    print("ciFAIR-10 loaded...")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    
    # remove ciFAIR-10.zip file that was downloaded
    import os
    zip_path = os.path.join('./data', ciFAIR10.filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)