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

# Legacy raw-frame state used by the old OpenCV/TensorRT path. The DeepStream
# server path does not populate this; DeepStream metadata updates cam_state and
# detection_log directly from the pad probe.
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

# SSE subscribers store (event_loop, asyncio.Queue) pairs so the DeepStream GLib
# thread can publish with loop.call_soon_threadsafe().
global_sse_subscribers = []
per_cam_sse_subscribers = {cid: [] for cid in cam_ids}
sse_sub_lock = threading.Lock()
