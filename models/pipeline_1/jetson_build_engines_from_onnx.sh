#!/usr/bin/env bash
set -euo pipefail

# Run this on the Jetson Nano.
# Do not build TensorRT .engine files on Windows, Orin, desktop GPU, or Colab
# and copy them to Jetson Nano.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

YOLO_ONNX="${YOLO_ONNX:-${PROJECT_ROOT}/models/pipeline_beta/yolov8n(1).onnx}"
YOLO_ENGINE="${YOLO_ENGINE:-${PROJECT_ROOT}/models/pipeline_beta/yolov8n_deepstream_nano_b5_fp32.engine}"
BRAND_ONNX="${BRAND_ONNX:-${SCRIPT_DIR}/exports/efficientnetb0_brand_opset12.onnx}"
BRAND_ENGINE="${BRAND_ENGINE:-${SCRIPT_DIR}/exports/efficientnetb0_brand_fp32.engine}"
YOLO_BATCH="${YOLO_BATCH:-5}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
BRAND_IMGSZ="${BRAND_IMGSZ:-224}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"
USE_FP16="${USE_FP16:-0}"

TRT_PRECISION_ARGS=()
if [ "${USE_FP16}" = "1" ]; then
  TRT_PRECISION_ARGS+=(--fp16)
fi

mkdir -p "$(dirname "${YOLO_ENGINE}")" "$(dirname "${BRAND_ENGINE}")"

echo "[1/2] Build YOLO detector engine"
trtexec \
  --onnx="${YOLO_ONNX}" \
  --saveEngine="${YOLO_ENGINE}" \
  --workspace="${WORKSPACE_MB}" \
  --minShapes=images:${YOLO_BATCH}x3x${YOLO_IMGSZ}x${YOLO_IMGSZ} \
  --optShapes=images:${YOLO_BATCH}x3x${YOLO_IMGSZ}x${YOLO_IMGSZ} \
  --maxShapes=images:${YOLO_BATCH}x3x${YOLO_IMGSZ}x${YOLO_IMGSZ} \
  "${TRT_PRECISION_ARGS[@]}"

echo "[2/2] Build EfficientNet-B0 brand engine"
trtexec \
  --onnx="${BRAND_ONNX}" \
  --saveEngine="${BRAND_ENGINE}" \
  --workspace="${WORKSPACE_MB}" \
  --minShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ} \
  --optShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ} \
  --maxShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ} \
  "${TRT_PRECISION_ARGS[@]}"

echo "Done:"
echo "  ${YOLO_ENGINE}"
echo "  ${BRAND_ENGINE}"
