#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DARKNET_DIR="${ROOT_DIR}/darknet"

mkdir -p "${DARKNET_DIR}"

download() {
  local url="$1"
  local dst="$2"

  if [[ -s "${dst}" ]]; then
    echo "[SKIP] ${dst} already exists"
    return
  fi

  echo "[GET] ${url}"
  curl -L --fail --retry 3 --output "${dst}" "${url}"
}

download \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" \
  "${DARKNET_DIR}/yolov4-tiny.cfg"

download \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" \
  "${DARKNET_DIR}/coco.names"

download \
  "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" \
  "${DARKNET_DIR}/yolov4-tiny.weights"

echo "[OK] YOLOv4-tiny Darknet assets are in ${DARKNET_DIR}"

