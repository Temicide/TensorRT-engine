from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MISSING_LABEL = -1


@dataclass
class TrainStats:
    loss: float
    brand_acc: float
    color_acc: float
    brand_count: int
    color_count: int


class SingleTaskFolderDataset(Dataset):
    """
    Reads split/class_name/image files for one task.

    For brand samples, color target is -1. For color samples, brand target is -1.
    CrossEntropyLoss is computed only where the target is not -1.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        task: str,
        class_to_idx: dict[str, int],
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.task = task
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
        image_path, class_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        brand_target = class_idx if self.task == "brand" else MISSING_LABEL
        color_target = class_idx if self.task == "color" else MISSING_LABEL
        return image, torch.tensor(brand_target), torch.tensor(color_target)


class EfficientNetB0MultiTask(nn.Module):
    def __init__(self, num_brands: int, num_colors: int, pretrained: bool):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.brand_head = nn.Linear(in_features, num_brands)
        self.color_head = nn.Linear(in_features, num_colors)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.brand_head(features), self.color_head(features)


def find_classes(root: Path, split: str) -> list[str]:
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
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
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


def build_dataset(args: argparse.Namespace, split: str, transform) -> tuple[Dataset, list[str], list[str]]:
    brand_root = Path(args.brand_root)
    color_root = Path(args.color_root)
    brand_classes = find_classes(brand_root, "train")
    color_classes = find_classes(color_root, "train")
    brand_to_idx = {name: idx for idx, name in enumerate(brand_classes)}
    color_to_idx = {name: idx for idx, name in enumerate(color_classes)}

    parts: list[Dataset] = []
    if (brand_root / split).exists():
        parts.append(SingleTaskFolderDataset(brand_root, split, "brand", brand_to_idx, transform))
    if (color_root / split).exists():
        parts.append(SingleTaskFolderDataset(color_root, split, "color", color_to_idx, transform))
    if not parts:
        raise FileNotFoundError(f"Neither brand nor color dataset has split: {split}")

    return ConcatDataset(parts), brand_classes, color_classes


def multitask_loss(
    brand_logits: torch.Tensor,
    color_logits: torch.Tensor,
    brand_targets: torch.Tensor,
    color_targets: torch.Tensor,
    brand_weight: float,
    color_weight: float,
) -> torch.Tensor:
    losses = []
    brand_mask = brand_targets != MISSING_LABEL
    color_mask = color_targets != MISSING_LABEL

    if brand_mask.any():
        losses.append(brand_weight * nn.functional.cross_entropy(brand_logits[brand_mask], brand_targets[brand_mask]))
    if color_mask.any():
        losses.append(color_weight * nn.functional.cross_entropy(color_logits[color_mask], color_targets[color_mask]))
    if not losses:
        return brand_logits.sum() * 0.0
    return sum(losses)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    mask = targets != MISSING_LABEL
    if not mask.any():
        return 0, 0
    pred = logits[mask].argmax(dim=1)
    correct = int((pred == targets[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    brand_weight: float,
    color_weight: float,
    amp: bool,
) -> TrainStats:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    brand_correct = brand_total = 0
    color_correct = color_total = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for images, brand_targets, color_targets in loader:
        images = images.to(device, non_blocking=True)
        brand_targets = brand_targets.to(device, non_blocking=True).long()
        color_targets = color_targets.to(device, non_blocking=True).long()

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                brand_logits, color_logits = model(images)
                loss = multitask_loss(
                    brand_logits,
                    color_logits,
                    brand_targets,
                    color_targets,
                    brand_weight,
                    color_weight,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        correct, total = accuracy(brand_logits.detach(), brand_targets)
        brand_correct += correct
        brand_total += total
        correct, total = accuracy(color_logits.detach(), color_targets)
        color_correct += correct
        color_total += total

    return TrainStats(
        loss=total_loss / max(total_items, 1),
        brand_acc=brand_correct / brand_total if brand_total else 0.0,
        color_acc=color_correct / color_total if color_total else 0.0,
        brand_count=brand_total,
        color_count=color_total,
    )


def save_checkpoint(
    output: Path,
    model: nn.Module,
    brand_classes: list[str],
    color_classes: list[str],
    args: argparse.Namespace,
    epoch: int,
    val_stats: TrainStats,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "brand_classes": brand_classes,
        "color_classes": color_classes,
        "epoch": epoch,
        "val_stats": asdict(val_stats),
        "args": vars(args),
    }
    torch.save(checkpoint, output)

    labels_path = output.with_suffix(".labels.json")
    labels_path.write_text(
        json.dumps(
            {"brand_classes": brand_classes, "color_classes": color_classes},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 with two heads: brand and color.")
    parser.add_argument("--brand-root", default="Car Brand Classification Dataset")
    parser.add_argument("--color-root", default="colordataset")
    parser.add_argument("--output", default="runs/efficientnetb0_mtl/best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--brand-weight", type=float, default=1.0)
    parser.add_argument("--color-weight", type=float, default=1.0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained EfficientNet-B0 weights.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Small debug subset.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Small debug subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, brand_classes, color_classes = build_dataset(args, "train", build_transforms(args.imgsz, True))
    val_split = "val" if (Path(args.brand_root) / "val").exists() or (Path(args.color_root) / "val").exists() else "test"
    val_dataset, _, _ = build_dataset(args, val_split, build_transforms(args.imgsz, False))

    train_dataset = maybe_limit(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_limit(val_dataset, args.max_val_samples, args.seed)

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

    model = EfficientNetB0MultiTask(len(brand_classes), len(color_classes), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device: {device}")
    print(f"brand classes: {len(brand_classes)} | color classes: {len(color_classes)}")
    print(f"train samples: {len(train_dataset)} | {val_split} samples: {len(val_dataset)}")

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.brand_weight,
            args.color_weight,
            args.amp,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                device,
                args.brand_weight,
                args.color_weight,
                args.amp,
            )

        score = (val_stats.brand_acc + val_stats.color_acc) / 2.0
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_stats.loss:.4f} "
            f"train_brand_acc={train_stats.brand_acc:.4f} "
            f"train_color_acc={train_stats.color_acc:.4f} "
            f"val_loss={val_stats.loss:.4f} "
            f"val_brand_acc={val_stats.brand_acc:.4f} "
            f"val_color_acc={val_stats.color_acc:.4f}"
        )

        if score > best_score:
            best_score = score
            save_checkpoint(Path(args.output), model, brand_classes, color_classes, args, epoch, val_stats)
            print(f"saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
