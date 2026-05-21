#!/usr/bin/env bash
set -euo pipefail

echo "[opencv] Removing PyPI OpenCV wheels that can shadow JetPack OpenCV..."
python3 -m pip uninstall -y \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless || true

echo "[opencv] Installing Jetson/Ubuntu OpenCV and GStreamer runtime packages..."
sudo apt update
sudo apt install -y \
  python3-opencv \
  python3-numpy \
  libopencv-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly

echo "[opencv] Verifying cv2 and GStreamer support..."
python3 - <<'PY'
import cv2

print("cv2 path:", cv2.__file__)
print("cv2 version:", cv2.__version__)

info = cv2.getBuildInformation()
gst_lines = [line.strip() for line in info.splitlines() if "GStreamer" in line]
if gst_lines:
    for line in gst_lines:
        print(line)
else:
    print("GStreamer: not found in cv2 build information")

if not any("GStreamer" in line and "YES" in line for line in gst_lines):
    raise SystemExit(
        "OpenCV imported, but this cv2 build does not report GStreamer=YES. "
        "Make sure your venv was created with --system-site-packages and that "
        "no pip OpenCV wheel is still shadowing /usr/lib/python3/dist-packages."
    )
PY

echo "[opencv] OK: OpenCV imports and reports GStreamer support."
