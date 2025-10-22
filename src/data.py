from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision import datasets
import os


def build_transforms(img_size: int = 64):
    tf_train = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToTensor(),
        v2.Normalize([0.5]*3, [0.5*3]),
    ])

    tf_val = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToTensor(),
        v2.Normalize([0.5]*3, [0.5*3])
    ])

    return tf_train, tf_val

def _count_per_class(dataset: datasets.ImageFolder):
    counts = [0]*len(dataset.classes)
    for _, y in dataset.samples:
        counts[y] += 1
    return counts

def create_dataloaders(
    data_root: str | Path = "data",
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
    use_weighted_sampling: bool = False,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    tf_train, tf_val = build_transforms(img_size)

    train_data = datasets.ImageFolder(str(train_dir), transform=tf_train)
    val_data = datasets.ImageFolder(str(val_dir), transform=tf_val)
    classes = train_data.classes


    if num_workers is None:
        cpu = os.cpu_count() or 4
        num_workers = min(8, max(2, cpu // 2))


    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # Class balancing
    train_sampler = None
    if use_weighted_sampling:
        counts = _count_per_class(train_data)
        # Weight per class = 1 / count
        class_weights = torch.tensor([1.0 / max(c, 1) for c in counts], dtype=torch.float)
        # Weight per sample
        sample_weights = [class_weights[y] for _, y in train_data.samples]
        train_sampler = WeightedRandomSampler(
            weights = sample_weights,
            num_samples = len(sample_weights),
            replacement = True
        )
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=max(256, batch_size*2),
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    return train_loader, val_loader, classes
    


    

