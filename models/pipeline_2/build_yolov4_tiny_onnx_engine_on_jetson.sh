#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${ONNX_PATH:-${ROOT_DIR}/exports/yolov4_tiny_416_raw.onnx}"
ENGINE_PATH="${ENGINE_PATH:-${ROOT_DIR}/exports/yolov4_tiny_416_raw_fp16.engine}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

TRTEXEC=""
for candidate in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec /usr/local/bin/trtexec; do
  if [[ -x "${candidate}" ]]; then
    TRTEXEC="${candidate}"
    break
  fi
done

if [[ -z "${TRTEXEC}" ]]; then
  echo "[ERROR] trtexec not found. Install TensorRT or add trtexec to PATH." >&2
  exit 1
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[ERROR] ONNX file not found: ${ONNX_PATH}" >&2
  echo "Run export_yolov4_tiny_onnx.py first." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --workspace="${WORKSPACE_MB}" \
  --fp16

echo "[OK] TensorRT engine created: ${ENGINE_PATH}"

