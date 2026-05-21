#!/usr/bin/env python3
"""Export Darknet YOLOv4-tiny cfg/weights to raw-head ONNX.

The exported ONNX model returns the raw YOLO feature maps. It intentionally does
not include Darknet YOLO decoding or NMS, which keeps the graph simple for
TensorRT on Jetson Nano.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class EmptyLayer(nn.Module):
    def forward(self, x):  # pragma: no cover - route/yolo handled in Darknet.forward
        return x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def parse_cfg(cfg_path: Path) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for raw_line in cfg_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current:
                blocks.append(current)
            current = {"type": line[1:-1].strip()}
            continue
        if "=" not in line:
            raise ValueError(f"Invalid cfg line in {cfg_path}: {raw_line}")
        key, value = line.split("=", 1)
        current[key.strip()] = value.strip()

    if current:
        blocks.append(current)

    if not blocks or blocks[0].get("type") != "net":
        raise ValueError(f"Expected first cfg block to be [net]: {cfg_path}")

    return blocks


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_activation(name: str) -> nn.Module:
    if name == "linear":
        return nn.Identity()
    if name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "mish":
        return Mish()
    if name == "logistic":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported Darknet activation: {name}")


class DarknetTiny(nn.Module):
    def __init__(self, blocks: List[Dict[str, str]]):
        super().__init__()
        self.net_info = blocks[0]
        self.module_defs = blocks[1:]
        self.module_list, self.output_filters = self._create_modules(self.module_defs)

    def _create_modules(
        self, module_defs: List[Dict[str, str]]
    ) -> Tuple[nn.ModuleList, List[int]]:
        module_list = nn.ModuleList()
        output_filters: List[int] = []
        prev_filters = int(self.net_info.get("channels", 3))

        for index, block in enumerate(module_defs):
            block_type = block["type"]

            if block_type == "convolutional":
                filters = int(block["filters"])
                kernel_size = int(block["size"])
                stride = int(block.get("stride", 1))
                pad_flag = int(block.get("pad", 0))
                padding = (kernel_size - 1) // 2 if pad_flag else 0
                batch_normalize = int(block.get("batch_normalize", 0))

                layers: List[nn.Module] = [
                    nn.Conv2d(
                        prev_filters,
                        filters,
                        kernel_size,
                        stride,
                        padding,
                        bias=not batch_normalize,
                    )
                ]
                if batch_normalize:
                    layers.append(nn.BatchNorm2d(filters))
                layers.append(make_activation(block.get("activation", "linear")))

                module_list.append(nn.Sequential(*layers))
                prev_filters = filters
                output_filters.append(filters)

            elif block_type == "maxpool":
                kernel_size = int(block["size"])
                stride = int(block["stride"])
                if kernel_size == 2 and stride == 1:
                    module_list.append(nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(2, 1, 0)))
                else:
                    padding = (kernel_size - 1) // 2
                    module_list.append(nn.MaxPool2d(kernel_size, stride, padding))
                output_filters.append(prev_filters)

            elif block_type == "upsample":
                stride = int(block["stride"])
                module_list.append(nn.Upsample(scale_factor=stride, mode="nearest"))
                output_filters.append(prev_filters)

            elif block_type == "route":
                layers = parse_int_list(block["layers"])
                filters = 0
                for layer_i in layers:
                    resolved = layer_i if layer_i >= 0 else index + layer_i
                    filters += output_filters[resolved]
                module_list.append(EmptyLayer())
                prev_filters = filters
                output_filters.append(filters)

            elif block_type == "shortcut":
                module_list.append(EmptyLayer())
                output_filters.append(prev_filters)

            elif block_type == "yolo":
                module_list.append(EmptyLayer())
                output_filters.append(prev_filters)

            else:
                raise ValueError(f"Unsupported Darknet layer [{block_type}] at index {index}")

        return module_list, output_filters

    def forward(self, x):
        outputs = []
        detections = []

        for index, (block, module) in enumerate(zip(self.module_defs, self.module_list)):
            block_type = block["type"]

            if block_type in ("convolutional", "maxpool", "upsample"):
                x = module(x)
            elif block_type == "route":
                layers = parse_int_list(block["layers"])
                tensors = []
                for layer_i in layers:
                    resolved = layer_i if layer_i >= 0 else index + layer_i
                    tensors.append(outputs[resolved])
                x = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)
            elif block_type == "shortcut":
                from_idx = int(block["from"])
                resolved = from_idx if from_idx >= 0 else index + from_idx
                x = outputs[-1] + outputs[resolved]
                activation = block.get("activation", "linear")
                if activation != "linear":
                    x = make_activation(activation)(x)
            elif block_type == "yolo":
                detections.append(outputs[-1])
            else:
                raise RuntimeError(f"Unhandled layer [{block_type}]")

            outputs.append(x)

        return tuple(detections)


def load_darknet_weights(model: DarknetTiny, weights_path: Path) -> None:
    with weights_path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=5)
        if header.size != 5:
            raise ValueError(f"Invalid Darknet weights header: {weights_path}")
        weights = np.fromfile(handle, dtype=np.float32)

    ptr = 0
    for block, module in zip(model.module_defs, model.module_list):
        if block["type"] != "convolutional":
            continue

        conv = module[0]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("Expected Conv2d at start of convolutional module")

        batch_normalize = int(block.get("batch_normalize", 0))
        if batch_normalize:
            bn = module[1]
            if not isinstance(bn, nn.BatchNorm2d):
                raise TypeError("Expected BatchNorm2d after normalized convolution")

            num_bn = bn.bias.numel()
            for target in (bn.bias, bn.weight, bn.running_mean, bn.running_var):
                next_ptr = ptr + num_bn
                if next_ptr > weights.size:
                    raise ValueError(f"Weights ended while reading batch norm for {weights_path}")
                target.data.copy_(torch.from_numpy(weights[ptr:next_ptr]).view_as(target))
                ptr = next_ptr
        else:
            num_bias = conv.bias.numel()
            next_ptr = ptr + num_bias
            if next_ptr > weights.size:
                raise ValueError(f"Weights ended while reading conv bias for {weights_path}")
            conv.bias.data.copy_(torch.from_numpy(weights[ptr:next_ptr]).view_as(conv.bias))
            ptr = next_ptr

        num_weights = conv.weight.numel()
        next_ptr = ptr + num_weights
        if next_ptr > weights.size:
            raise ValueError(f"Weights ended while reading conv weights for {weights_path}")
        conv.weight.data.copy_(torch.from_numpy(weights[ptr:next_ptr]).view_as(conv.weight))
        ptr = next_ptr

    if ptr != weights.size:
        unused = weights.size - ptr
        print(f"[WARN] {unused} unused float32 weights remain after loading", file=sys.stderr)


def export_onnx(
    cfg_path: Path,
    weights_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    simplify_names: bool,
) -> None:
    blocks = parse_cfg(cfg_path)
    model = DarknetTiny(blocks)
    load_darknet_weights(model, weights_path)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    output_names = ["detect_0", "detect_1"] if simplify_names else None

    with torch.no_grad():
        heads = model(dummy)
        if len(heads) != 2:
            raise RuntimeError(f"Expected 2 YOLOv4-tiny detection heads, got {len(heads)}")
        for name, tensor in zip(["detect_0", "detect_1"], heads):
            print(f"[INFO] {name}: {tuple(tensor.shape)}")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
    )

    print(f"[OK] ONNX exported: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Darknet YOLOv4-tiny to ONNX raw heads.")
    parser.add_argument("--cfg", default="darknet/yolov4-tiny.cfg", help="Path to yolov4-tiny.cfg")
    parser.add_argument("--weights", default="darknet/yolov4-tiny.weights", help="Path to yolov4-tiny.weights")
    parser.add_argument("--output", default="exports/yolov4_tiny_416_raw.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=416, help="Static square input size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset for TensorRT compatibility")
    parser.add_argument(
        "--default-output-names",
        action="store_true",
        help="Let PyTorch choose output names instead of detect_0/detect_1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.cfg)
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not cfg_path.is_file():
        print(f"[ERROR] cfg not found: {cfg_path}", file=sys.stderr)
        return 1
    if not weights_path.is_file():
        print(f"[ERROR] weights not found: {weights_path}", file=sys.stderr)
        return 1
    if args.imgsz <= 0:
        print("[ERROR] --imgsz must be positive", file=sys.stderr)
        return 1

    try:
        export_onnx(
            cfg_path=cfg_path,
            weights_path=weights_path,
            output_path=output_path,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify_names=not args.default_output_names,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

