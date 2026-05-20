# -*- coding: utf-8 -*-
"""Convert an ONNX model to a TensorRT engine on Jetson Nano.

TensorRT engine files are hardware-specific. Run this script on the Jetson
Nano that will use the generated .engine file.

Examples:
    python3 convert-onnx-to-engine.py
    python3 convert-onnx-to-engine.py --onnx yolov8n.onnx
    python3 convert-onnx-to-engine.py --onnx yolov8n.onnx --engine yolov8n.engine
    python3 convert-onnx-to-engine.py --onnx model.onnx --fp16
    python3 convert-onnx-to-engine.py --onnx model.onnx --force
    python3 convert-onnx-to-engine.py \
        --onnx dynamic.onnx \
        --input-name images \
        --min-shape 1x3x320x320 \
        --opt-shape 1x3x640x640 \
        --max-shape 1x3x640x640
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


TRTEXEC_CANDIDATES = (
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/bin/trtexec",
    "/usr/local/bin/trtexec",
)


def find_trtexec() -> Optional[str]:
    """Return a usable trtexec path, preferring common Jetson locations."""
    for candidate in TRTEXEC_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    found = shutil.which("trtexec")
    if found:
        return found

    return None


def prompt_for_onnx_path() -> Path:
    """Prompt until the user enters an existing file path."""
    print("No --onnx argument provided. Enter the path to your ONNX file.")

    while True:
        raw = input("ONNX path: ").strip()
        if not raw:
            print("Path cannot be empty. Try again.")
            continue

        path = Path(raw).expanduser()
        if not path.exists():
            print("File not found: {0}. Try again.".format(path))
            continue
        if not path.is_file():
            print("Path is not a file: {0}. Try again.".format(path))
            continue

        if path.suffix.lower() != ".onnx":
            print("[WARN] File does not end with .onnx: {0}".format(path))

        return path


def validate_onnx_path(path: Path) -> None:
    """Validate that the input ONNX path exists and is a file."""
    if not path.exists():
        raise ValueError("ONNX file not found: {0}".format(path))
    if not path.is_file():
        raise ValueError("ONNX path is not a file: {0}".format(path))


def validate_shape(shape: str, label: str) -> None:
    """Validate trtexec shape text such as 1x3x640x640."""
    parts = shape.lower().split("x")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("{0} must look like 1x3x640x640".format(label))

    dims = [int(part) for part in parts]
    if any(dim <= 0 for dim in dims):
        raise ValueError("{0} dimensions must be positive integers".format(label))


def dynamic_shapes_requested(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.input_name,
            args.min_shape,
            args.opt_shape,
            args.max_shape,
        )
    )


def validate_dynamic_shape_args(args: argparse.Namespace) -> None:
    """Require the complete dynamic-shape argument set when any part is used."""
    if not dynamic_shapes_requested(args):
        return

    missing = []
    if not args.input_name:
        missing.append("--input-name")
    if not args.min_shape:
        missing.append("--min-shape")
    if not args.opt_shape:
        missing.append("--opt-shape")
    if not args.max_shape:
        missing.append("--max-shape")

    if missing:
        raise ValueError(
            "Dynamic shape conversion requires all of: "
            "--input-name, --min-shape, --opt-shape, --max-shape. "
            "Missing: {0}".format(", ".join(missing))
        )

    validate_shape(args.min_shape, "--min-shape")
    validate_shape(args.opt_shape, "--opt-shape")
    validate_shape(args.max_shape, "--max-shape")


def default_engine_path(onnx_path: Path, fp16: bool) -> Path:
    suffix = "_fp16.engine" if fp16 else "_fp32.engine"
    return onnx_path.parent / (onnx_path.stem + suffix)


def build_trtexec_command(
    trtexec_path: str,
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    workspace_mb: int,
    input_name: Optional[str],
    min_shape: Optional[str],
    opt_shape: Optional[str],
    max_shape: Optional[str],
) -> List[str]:
    """Build the trtexec command without guessing static model shapes."""
    cmd = [
        trtexec_path,
        "--onnx={0}".format(onnx_path),
        "--saveEngine={0}".format(engine_path),
        "--workspace={0}".format(workspace_mb),
    ]

    if fp16:
        cmd.append("--fp16")

    if input_name:
        cmd.extend(
            [
                "--minShapes={0}:{1}".format(input_name, min_shape),
                "--optShapes={0}:{1}".format(input_name, opt_shape),
                "--maxShapes={0}:{1}".format(input_name, max_shape),
            ]
        )

    return cmd


def run_trtexec(cmd: List[str]) -> int:
    """Run trtexec with live stdout/stderr streaming."""
    print("[INFO] Running: {0}".format(" ".join(cmd)))
    process = subprocess.Popen(cmd)
    return process.wait()


def convert_onnx_to_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    workspace_mb: int,
    force: bool,
    input_name: Optional[str],
    min_shape: Optional[str],
    opt_shape: Optional[str],
    max_shape: Optional[str],
) -> Path:
    """Convert ONNX to TensorRT engine using trtexec."""
    validate_onnx_path(onnx_path)

    if workspace_mb <= 0:
        raise ValueError("--workspace must be a positive integer")

    if engine_path.exists() and not force:
        raise ValueError(
            "Engine already exists: {0}. Use --force to overwrite it.".format(
                engine_path
            )
        )
    if engine_path.exists() and force:
        if not engine_path.is_file():
            raise ValueError(
                "Engine path exists and is not a file: {0}".format(engine_path)
            )
        engine_path.unlink()

    trtexec_path = find_trtexec()
    if trtexec_path is None:
        raise RuntimeError(
            "Could not find trtexec. On Jetson, check that TensorRT is installed "
            "and that /usr/src/tensorrt/bin/trtexec exists, or add trtexec to PATH."
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_trtexec_command(
        trtexec_path=trtexec_path,
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=fp16,
        workspace_mb=workspace_mb,
        input_name=input_name,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    rc = run_trtexec(cmd)
    if rc != 0:
        raise RuntimeError(
            "trtexec failed with exit code {0}. If TensorRT reports unsupported "
            "ONNX ops, parser errors, or opset issues, re-export the model for "
            "the TensorRT version installed on this Jetson instead of rewriting "
            "the ONNX on-device.".format(rc)
        )

    if not engine_path.exists():
        raise RuntimeError(
            "trtexec exited successfully but did not create: {0}".format(engine_path)
        )
    if engine_path.stat().st_size == 0:
        raise RuntimeError(
            "trtexec created an empty engine file: {0}".format(engine_path)
        )

    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX file to a TensorRT .engine file on Jetson Nano. Defaults to FP32."
    )
    parser.add_argument(
        "--onnx",
        "-i",
        default=None,
        help="Path to input ONNX file. If omitted, the script prompts for it.",
    )
    parser.add_argument(
        "--engine",
        "-o",
        default=None,
        help="Path to output .engine file. Defaults to <onnx_stem>_fp32.engine.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Build an FP16 engine. Test only after FP32 works on Jetson Nano.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Compatibility flag; FP32 is already the default.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="TensorRT workspace size in MB. Default: 1024.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output engine if it already exists.",
    )
    parser.add_argument(
        "--input-name",
        default=None,
        help="Dynamic-shape input tensor name, for example: images.",
    )
    parser.add_argument(
        "--min-shape",
        default=None,
        help="Dynamic minimum shape, for example: 1x3x320x320.",
    )
    parser.add_argument(
        "--opt-shape",
        default=None,
        help="Dynamic optimum shape, for example: 1x3x640x640.",
    )
    parser.add_argument(
        "--max-shape",
        default=None,
        help="Dynamic maximum shape, for example: 1x3x640x640.",
    )

    args = parser.parse_args()
    try:
        validate_dynamic_shape_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main() -> int:
    args = parse_args()

    if args.fp16 and args.fp32:
        print("[ERROR] Use either --fp16 or --fp32, not both.", file=sys.stderr)
        return 1

    fp16 = bool(args.fp16)
    onnx_path = Path(args.onnx).expanduser() if args.onnx else prompt_for_onnx_path()
    engine_path = (
        Path(args.engine).expanduser()
        if args.engine
        else default_engine_path(onnx_path, fp16)
    )

    try:
        validate_onnx_path(onnx_path)
        print("[INFO] Input ONNX: {0}".format(onnx_path))
        print("[INFO] Output engine: {0}".format(engine_path))
        print(
            "[INFO] Precision: {0} | workspace: {1} MB".format(
                "FP16" if fp16 else "FP32",
                args.workspace,
            )
        )
        if dynamic_shapes_requested(args):
            print(
                "[INFO] Dynamic shapes: {0} min={1} opt={2} max={3}".format(
                    args.input_name,
                    args.min_shape,
                    args.opt_shape,
                    args.max_shape,
                )
            )

        result = convert_onnx_to_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=fp16,
            workspace_mb=args.workspace,
            force=args.force,
            input_name=args.input_name,
            min_shape=args.min_shape,
            opt_shape=args.opt_shape,
            max_shape=args.max_shape,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print("[ERROR] {0}".format(exc), file=sys.stderr)
        return 1

    print("[INFO] Engine created: {0}".format(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
