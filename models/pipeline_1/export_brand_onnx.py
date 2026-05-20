from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_efficientnetb0_brand import EfficientNetB0Brand


def export_brand_onnx(
    checkpoint_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    dynamic: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    brand_classes = checkpoint["brand_classes"]

    model = EfficientNetB0Brand(num_brands=len(brand_classes), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, imgsz, imgsz)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch"},
            "brand_logits": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["brand_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    labels_path = output_path.with_suffix(".labels.json")
    labels_path.write_text(
        json.dumps({"brand_classes": brand_classes, "imgsz": imgsz}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved ONNX: {output_path}")
    print(f"saved labels: {labels_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EfficientNet-B0 brand classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="runs/efficientnetb0_brand/best.pt")
    parser.add_argument("--output", default="exports/efficientnetb0_brand.onnx")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_brand_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
