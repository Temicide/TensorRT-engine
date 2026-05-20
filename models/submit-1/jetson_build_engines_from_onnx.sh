#!/usr/bin/env bash
set -euo pipefail

# Run this on the Jetson Nano.
# Do not build TensorRT .engine files on Windows and copy them to Jetson.

YOLO_ONNX="${YOLO_ONNX:-exports/yolo26n_opset12.onnx}"
YOLO_ENGINE="${YOLO_ENGINE:-exports/yolo26n_fp16.engine}"
BRAND_ONNX="${BRAND_ONNX:-exports/efficientnetb0_brand_opset12.onnx}"
BRAND_ENGINE="${BRAND_ENGINE:-exports/efficientnetb0_brand_fp16.engine}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
BRAND_IMGSZ="${BRAND_IMGSZ:-224}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

mkdir -p exports

echo "[1/2] Build YOLO26n detector engine"
trtexec \
  --onnx="${YOLO_ONNX}" \
  --saveEngine="${YOLO_ENGINE}" \
  --fp16 \
  --workspace="${WORKSPACE_MB}" \
  --minShapes=images:1x3x${YOLO_IMGSZ}x${YOLO_IMGSZ} \
  --optShapes=images:1x3x${YOLO_IMGSZ}x${YOLO_IMGSZ} \
  --maxShapes=images:1x3x${YOLO_IMGSZ}x${YOLO_IMGSZ}

echo "[2/2] Build EfficientNet-B0 brand engine"
trtexec \
  --onnx="${BRAND_ONNX}" \
  --saveEngine="${BRAND_ENGINE}" \
  --fp16 \
  --workspace="${WORKSPACE_MB}" \
  --minShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ} \
  --optShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ} \
  --maxShapes=images:1x3x${BRAND_IMGSZ}x${BRAND_IMGSZ}

echo "Done:"
echo "  ${YOLO_ENGINE}"
echo "  ${BRAND_ENGINE}"
