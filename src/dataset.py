"""
dataset.py
==========
Data loading for CIFAR-10 or Fashion-MNIST using torchvision. Downloads
automatically on first run (cached under data_dir). Which dataset is used
is controlled by configs/training_config.yaml -> data.dataset.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    is_fashion = dataset_name.lower() in ("fashion_mnist", "fashionmnist")

    base = []
    if is_fashion:
        base += [transforms.Grayscale(num_output_channels=3), transforms.Resize((32, 32))]

    if train and not is_fashion:
        base += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]

    mean, std = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    base += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(base)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    is_fashion = dataset_name.lower() in ("fashion_mnist", "fashionmnist")
    dataset_cls = datasets.FashionMNIST if is_fashion else datasets.CIFAR10

    train_dataset = dataset_cls(
        root=data_dir, train=True, download=True,
        transform=get_transforms(dataset_name, train=True),
    )
    val_dataset = dataset_cls(
        root=data_dir, train=False, download=True,
        transform=get_transforms(dataset_name, train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader