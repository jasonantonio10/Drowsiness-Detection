# src/inspect_batch.py
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from collections import Counter
from pathlib import Path
from data import create_dataloaders

# --- Transforms ---
tf_train = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Load dataset ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "train"

train_ds = datasets.ImageFolder(str(DATA_DIR), transform=tf_train)
classes = train_ds.classes
class_to_idx = train_ds.class_to_idx

print("Classes:", classes)
print("class_to_idx:", class_to_idx)

# --- Plot class distribution ---
def plot_class_distribution(ds, classes):
    counts = Counter([y for _, y in ds.samples])
    labels = [classes[i] for i in range(len(classes))]
    values = [counts[i] for i in range(len(classes))]

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Class Distribution (Train Set)")
    plt.ylabel("Number of Images")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

# --- Show batch ---
def show_batch(dl, classes, n=32):
    x, y = next(iter(dl))
    grid = make_grid((x[:n]*0.5 + 0.5), nrow=8)  # unnormalize for viewing
    labels = [classes[i] for i in y[:n]]
    print("Labels:", labels)

    plt.figure(figsize=(10,6))
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.title("Sample Batch")
    plt.show()

# --- Run ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("\n--- Class distribution ---")
    plot_class_distribution(train_ds, classes)

    print("\n--- Sample batch ---")
    train_loader, _, classes = create_dataloaders(
        data_root=ROOT / "data",
        img_size=64,
        batch_size=64,
        use_weighted_sampling=False
    )
    show_batch(train_loader, classes)
