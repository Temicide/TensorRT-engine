from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EpochStats:
    loss: float
    acc: float
    count: int


class BrandFolderDataset(Dataset):
    def __init__(self, root: str | Path, split: str, class_to_idx: dict[str, int], transform=None):
        self.root = Path(root)
        self.split = split
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((image_path, class_idx))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


class EfficientNetB0Brand(nn.Module):
    def __init__(self, num_brands: int, pretrained: bool):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.brand_head = nn.Linear(in_features, num_brands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.brand_head(self.backbone(x))


def find_classes(root: Path, split: str = "train") -> list[str]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under {split_dir}")
    return classes


def build_transforms(imgsz: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((imgsz + 32, imgsz + 32)),
                transforms.RandomResizedCrop(imgsz, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def maybe_limit(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch = images.size(0)
        total_loss += float(loss.item()) * batch
        pred = logits.detach().argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += batch

    return EpochStats(loss=total_loss / max(total, 1), acc=correct / max(total, 1), count=total)


def save_checkpoint(
    output: Path,
    model: nn.Module,
    brand_classes: list[str],
    args: argparse.Namespace,
    epoch: int,
    val_stats: EpochStats,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "efficientnet_b0_brand",
            "model": model.state_dict(),
            "brand_classes": brand_classes,
            "epoch": epoch,
            "val_stats": asdict(val_stats),
            "args": vars(args),
        },
        output,
    )
    output.with_suffix(".labels.json").write_text(
        json.dumps({"brand_classes": brand_classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for car brand classification only.")
    parser.add_argument("--brand-root", default="Car Brand Classification Dataset")
    parser.add_argument("--output", default="runs/efficientnetb0_brand/best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.brand_root)
    classes = find_classes(root)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    val_split = "val" if (root / "val").exists() else "test"

    train_dataset = BrandFolderDataset(root, "train", class_to_idx, build_transforms(args.imgsz, True))
    val_dataset = BrandFolderDataset(root, val_split, class_to_idx, build_transforms(args.imgsz, False))
    train_dataset = maybe_limit(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_limit(val_dataset, args.max_val_samples, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = EfficientNetB0Brand(len(classes), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device: {device}")
    print(f"brand classes: {len(classes)}")
    print(f"train samples: {len(train_dataset)} | {val_split} samples: {len(val_dataset)}")

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, args.amp)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, None, device, args.amp)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
            f"val_loss={val_stats.loss:.4f} val_acc={val_stats.acc:.4f}"
        )

        if val_stats.acc > best_acc:
            best_acc = val_stats.acc
            save_checkpoint(Path(args.output), model, classes, args, epoch, val_stats)
            print(f"saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
