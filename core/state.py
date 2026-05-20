import collections
import threading
from config import CONFIG

# ──────────────────────────── Shared state per camera ─────────────────────────
cam_ids = CONFIG.get("active_cameras") or list(CONFIG["cameras"].keys())

cam_state = {
    cid: {
        "frame_jpg": b"",
        "detections": [],
        "fps": 0.0,
        "frame_count": 0,
        "lock": threading.Lock(),
        "sse_queue": collections.deque(maxlen=50),   # SSE per camera
    }
    for cid in cam_ids
}

# Latest raw frames from camera threads.
# Camera threads only read RTSP frames and never touch CUDA/TensorRT.
# A single inference worker owns TensorRT calls. This is much more stable
# on Jetson Nano than calling one shared CUDA context from 5 camera threads.
latest_frame_state = {
    cid: {
        "frame": None,
        "frame_count": 0,
        "updated_at": 0.0,
        "lock": threading.Lock(),
    }
    for cid in cam_ids
}

# Central detection log (all cameras)
detection_log = []
detection_log_lock = threading.Lock()
MAX_LOG = 5000

# SSE queues: per-camera subscribers + global subscribers
global_sse_subscribers = []
per_cam_sse_subscribers = {cid: [] for cid in cam_ids}
sse_sub_lock = threading.Lock()