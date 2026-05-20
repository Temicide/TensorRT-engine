import collections
import threading
import time
import requests
import logging
from config import CONFIG

log = logging.getLogger("multicam")

# ─────────────────────────── External API push (circuit breaker + queue) ──────
_API_QUEUE      = collections.deque(maxlen=200)
_API_QUEUE_LOCK = threading.Lock()
_API_QUEUE_EVT  = threading.Event()

_cb = {"fails": 0, "open": False, "open_until": 0.0}
FAIL_THRESHOLD = 5
COOLDOWN_SEC   = 30


def _build_api_payloads(cam_id, record):
    out = []
    for det in record.get("detections", []):
        x1, y1, x2, y2 = det.get("bbox_xyxy", [0, 0, 0, 0])
        out.append({
            "device_name": CONFIG["device_name"],
            "data": {
                "timestamp": record["timestamp"],
                "type":      det["class_name"],
                "color":     "",
                "brand":     "",
                "x":         round(float(x1), 4),
                "y":         round(float(y1), 4),
                "width":     round(float(x2 - x1), 4),
                "height":    round(float(y2 - y1), 4),
                "camera_id": cam_id,
                "jetson_id": CONFIG["jetson_id"],
                "track_id":  str(det.get("track_id", "")),
            }
        })
    return out


def _api_worker():
    sess = requests.Session()
    while True:
        _API_QUEUE_EVT.wait(timeout=5)
        _API_QUEUE_EVT.clear()
        if _cb["open"]:
            if time.time() < _cb["open_until"]:
                continue
            _cb["open"] = False; _cb["fails"] = 0
            log.info("External API circuit CLOSED — retrying")
        while True:
            with _API_QUEUE_LOCK:
                if not _API_QUEUE: break
                cam_id, record = _API_QUEUE.popleft()
            for payload in _build_api_payloads(cam_id, record):
                try:
                    sess.post(CONFIG["external_api"], json=payload, timeout=2)
                    if _cb["fails"]:
                        log.info("External API OK — resetting fail count")
                    _cb["fails"] = 0
                except Exception as e:
                    _cb["fails"] += 1
                    if _cb["fails"] == FAIL_THRESHOLD:
                        _cb["open"] = True
                        _cb["open_until"] = time.time() + COOLDOWN_SEC
                        log.warning(
                            f"[{cam_id}] External API unreachable ({e}). "
                            f"Circuit OPEN — pausing {COOLDOWN_SEC}s."
                        )
                    if _cb["open"]: break
            if _cb["open"]: break


threading.Thread(target=_api_worker, daemon=True, name="api-worker").start()


def push_external_async(cam_id, record):
    if not record.get("detections"):
        return
    with _API_QUEUE_LOCK:
        _API_QUEUE.append((cam_id, record))
    _API_QUEUE_EVT.set()