from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

from ultralytics import YOLO  # noqa: E402


def normalize_weights_name(weights: str) -> str:
    return "yolo26n.pt" if weights == "yolov26n.pt" else weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO detector weights to ONNX for Jetson/TensorRT.")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(normalize_weights_name(args.weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(f"saved ONNX: {exported}")


if __name__ == "__main__":
    main()
