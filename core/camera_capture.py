import logging
import threading
import time

import cv2

from config import CONFIG
from core.state import latest_frame_state

log = logging.getLogger("multicam")

# Legacy OpenCV capture path. The production server no longer imports this
# module; RTSP ingest is handled by DeepStream rtspsrc/nvv4l2decoder.

# ─────────────────────────── Pipeline loop per camera ─────────────────────────
CAPTURE_OPEN_LOCK = threading.Lock()

def open_capture(cam_id, source):
    """Open RTSP capture safely on Jetson Nano.

    Important: do not let 5 camera threads create VideoCapture at the same time.
    Some OpenCV/GStreamer/FFmpeg builds on Jetson can segfault during concurrent RTSP open.
    """
    with CAPTURE_OPEN_LOCK:
        if source.startswith("rtsp://") and CONFIG.get("enable_gstreamer", False):
            gst_candidates = [
                (
                    f"rtspsrc location={source} protocols=tcp latency=200 drop-on-latency=true ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
                    "video/x-raw,format=BGRx ! videoconvert ! "
                    "video/x-raw,format=BGR ! appsink drop=1 sync=false max-buffers=1"
                ),
                (
                    f"rtspsrc location={source} protocols=tcp latency=200 drop-on-latency=true ! "
                    "rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! "
                    "video/x-raw,format=BGRx ! videoconvert ! "
                    "video/x-raw,format=BGR ! appsink drop=1 sync=false max-buffers=1"
                ),
                (
                    f"rtspsrc location={source} protocols=tcp latency=200 drop-on-latency=true ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                    "video/x-raw,format=BGR ! appsink drop=1 sync=false max-buffers=1"
                ),
            ]

            for gst in gst_candidates:
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    log.info(f"[{cam_id}] GStreamer capture OK")
                    return cap

            log.warning(f"[{cam_id}] GStreamer capture failed, falling back to OpenCV/FFmpeg")

        # Safer fallback. OPENCV_FFMPEG_CAPTURE_OPTIONS above requests RTSP over TCP.
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap


def capture_loop(cam_id, source):
    """Read frames only. This thread must not call TensorRT/CUDA."""
    frame_idx = 0

    while True:
        cap = open_capture(cam_id, source)
        if not cap.isOpened():
            log.error(f"[{cam_id}] Cannot open source: {source}, retrying in 5s...")
            time.sleep(5)
            continue

        log.info(f"[{cam_id}] Stream opened: {source}")

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning(f"[{cam_id}] Frame read failed, reconnecting...")
                break

            slot = latest_frame_state[cam_id]
            with slot["lock"]:
                # Keep only the newest frame. This prevents RTSP backlog and keeps latency low.
                slot["frame"] = frame
                slot["frame_count"] = frame_idx
                slot["updated_at"] = time.time()

            frame_idx += 1

        cap.release()
        time.sleep(2)
