import argparse
import os
import shutil
from pathlib import Path
from typing import List, Set


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

POSTPROCESS_OPS = {
    "BatchedNMS",
    "BatchedNMS_TRT",
    "EfficientNMS",
    "EfficientNMS_TRT",
    "Mod",
    "NonMaxSuppression",
    "TopK",
}

POSTPROCESS_OP_TOKENS = {
    b"\x22\x0aBatchedNMS": "BatchedNMS",
    b"\x22\x0eBatchedNMS_TRT": "BatchedNMS_TRT",
    b"\x22\x0cEfficientNMS": "EfficientNMS",
    b"\x22\x10EfficientNMS_TRT": "EfficientNMS_TRT",
    b"\x22\x03Mod": "Mod",
    b"\x22\x11NonMaxSuppression": "NonMaxSuppression",
    b"\x22\x04TopK": "TopK",
}


def normalize_weights_name(weights: str) -> str:
    return "yolo26n.pt" if weights == "yolov26n.pt" else weights


def resolve_input_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)

    local_path = ROOT / path
    if local_path.exists():
        return str(local_path)
    return str(path)


def default_output_path(weights: str, opset: int) -> Path:
    stem = Path(normalize_weights_name(weights)).stem
    return ROOT / "exports" / f"{stem}_opset{opset}.onnx"


def detect_postprocess_ops_with_onnx(onnx_path: Path) -> Set[str]:
    import onnx  # type: ignore

    model = onnx.load(str(onnx_path), load_external_data=False)
    detected = set()
    for node in model.graph.node:
        if node.op_type in POSTPROCESS_OPS or "NMS" in node.op_type:
            detected.add(node.op_type)
    return detected


def detect_postprocess_ops_from_bytes(onnx_path: Path) -> Set[str]:
    data = onnx_path.read_bytes()
    return {
        op_name
        for token, op_name in POSTPROCESS_OP_TOKENS.items()
        if token in data
    }


def detect_postprocess_ops(onnx_path: Path) -> List[str]:
    try:
        detected = detect_postprocess_ops_with_onnx(onnx_path)
    except Exception:
        detected = detect_postprocess_ops_from_bytes(onnx_path)
    return sorted(detected)


def move_exported_onnx(exported: str, output_path: Path) -> Path:
    exported_path = Path(exported).expanduser()
    if not exported_path.is_absolute():
        cwd_path = Path.cwd() / exported_path
        root_path = ROOT / exported_path
        exported_path = cwd_path if cwd_path.exists() else root_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() == output_path.resolve():
        return output_path

    if output_path.exists():
        output_path.unlink()
    shutil.move(str(exported_path), str(output_path))
    return output_path


def validate_deepstream_onnx(onnx_path: Path) -> None:
    detected = detect_postprocess_ops(onnx_path)
    if not detected:
        return

    ops = ", ".join(detected)
    raise RuntimeError(
        "Exported ONNX contains post-processing operators that are not suitable "
        f"for the Jetson Nano DeepStream-YOLO path: {ops}. Export raw YOLO "
        "detection outputs for DeepStream. Do not use --nms for this server."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO detector weights to ONNX for Jetson/TensorRT.")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path. Default: exports/<weights_stem>_opset<opset>.onnx.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include Ultralytics NMS/post-processing in the ONNX. Do not use for DeepStream nvinfer.",
    )
    parser.add_argument(
        "--skip-deepstream-onnx-check",
        action="store_true",
        help="Skip the DeepStream compatibility check for NMS/TopK/Mod operators.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # noqa: E402

    weights = normalize_weights_name(args.weights)
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else default_output_path(weights, args.opset)
    )
    if args.output and not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    model = YOLO(resolve_input_path(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        batch=args.batch,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=args.nms,
    )
    saved_path = move_exported_onnx(exported, output_path)
    if not args.nms and not args.skip_deepstream_onnx_check:
        validate_deepstream_onnx(saved_path)
    print(f"saved ONNX: {saved_path}")


if __name__ == "__main__":
    main()
