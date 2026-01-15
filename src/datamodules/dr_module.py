from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
from pathlib import Path
import cv2
import sys 

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.image_transforms import CLAHETransform, CropTransform


class RDDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class RDDatamodule(LightningDataModule):
    """
        DataModule for Diabetic Retinopathy Dataset
        In order to get more info about images the following transfromations are applied:
        - Circular Masking: to focus on the retinal area and eliminate background noise.
        - CLAHE (Contrast Limited Adaptive Histogram Equalization): to enhance the contrast of retinal images
    """
    
    def __init__(
        self,
        data_dir: Path = Path("./data/processed"),
        batch_size: int = 16,
        num_workers: int = 8,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

        self.train_transform = transforms.Compose([
            CropTransform(),
            transforms.Resize((224, 224)),
            CLAHETransform(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),      # losowe odbicie poziome
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.val_test_transform = transforms.Compose([
            CropTransform(),
            transforms.Resize((224, 224)),
            CLAHETransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def _print_class_distribution(self, name: str, samples: list[tuple[Path, int]]):
        counter = Counter(label for _, label in samples)
        print(f"\n{name} class distribution:")
        for cls, count in sorted(counter.items()):
            print(f"  Class {cls}: {count}")

    def setup(self, stage=None):
        samples = []
        class_to_idx = {}

        for idx, cls_folder in enumerate(sorted(self.data_dir.iterdir())):
            if cls_folder.is_dir():
                class_to_idx[cls_folder.name] = idx
                for img_path in cls_folder.iterdir():
                    if img_path.suffix.lower() == ".png":
                        samples.append((img_path, idx))

        train_samples, temp_samples = train_test_split(
            samples, test_size=self.val_split + self.test_split, stratify=[s[1] for s in samples]
        )

        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=self.test_split / (self.val_split + self.test_split),
            stratify=[s[1] for s in temp_samples],
        )

        self._print_class_distribution("TRAIN", train_samples)
        self._print_class_distribution("VAL", val_samples)
        self._print_class_distribution("TEST", test_samples)

        self.train_dataset = RDDataset(train_samples, transform=self.train_transform)
        self.val_dataset = RDDataset(val_samples, transform=self.val_test_transform)
        self.test_dataset = RDDataset(test_samples, transform=self.val_test_transform)

        print(f"\nTotal samples\n Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

def test_datamodule():
    data_module = RDDatamodule()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))

    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels: {labels.tolist()}")

    classes_to_show = [0, 1, 2, 3, 4]
    class_samples = {}

    for img, label in zip(images, labels):
        label = label.item()
        if label in classes_to_show and label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == len(classes_to_show):
            break

    fig, axes = plt.subplots(4, len(classes_to_show), figsize=(4 * len(classes_to_show), 8))

    for col, cls in enumerate(classes_to_show):
        if cls not in class_samples:
            print(f"Brak klasy {cls} w batchu")
            continue

        img = class_samples[cls].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Class {cls}")
        axes[0, col].axis("off")

        for ch in range(3):
            axes[ch + 1, col].imshow(img[:, :, ch], cmap="gray")
            axes[ch + 1, col].axis("off")

    plt.tight_layout()
    plt.show()
    # jak sobie rzucicie okiem na skałodwe RGB to najwięcej informacje jest w kanale G i B, w kanale R jest mniej informacji
    # o żyłkach siatkówki

if __name__ == "__main__":
    test_datamodule()
    
