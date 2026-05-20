# -*- coding: utf-8 -*-
"""
Multi-Camera YOLO TensorRT Jetson Pipeline - safe capture
5 RTSP streams -> TensorRT YOLO engine -> per-camera MJPEG + SSE + log
All detection logs POST to: http://10.0.11.153:8080/api/v1/raw_data

Endpoints:
  /cam{1-5}/live        -> MJPEG dashboard per camera
  /cam{1-5}/video       -> MJPEG stream per camera
  /cam{1-5}/stream      -> SSE per camera
  /cam{1-5}/loglive     -> redirect to central log dashboard
  /log/live             -> central log dashboard (all cameras)
  /log/stream           -> SSE for all cameras combined
  /detections           -> query log (?limit=50&class_name=person&camera_id=cam1)
"""

import threading
import time
import json
import logging
import collections
import uuid
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

# Make OpenCV/FFmpeg RTSP capture more stable on Jetson Nano.
# These options must be set before cv2 opens any VideoCapture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;500000")

import cv2
import numpy as np

# Avoid OpenCV spawning many internal CPU threads on Jetson Nano.
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# TensorRT inference on Jetson GPU
# Required on Jetson: python3-libnvinfer + pycuda
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # Do NOT use pycuda.autoinit in this multi-threaded server.
    # We create and push/pop the CUDA context explicitly.
except Exception as e:
    trt = None
    cuda = None
    TRT_IMPORT_ERROR = e
else:
    TRT_IMPORT_ERROR = None
import requests
from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import uvicorn

# ─────────────────────────── CONFIG (edit only here) ──────────────────────────
CONFIG = {
    # TensorRT engine built for this exact Jetson / TensorRT / CUDA version.
    # Example: /usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
    "model_path":     "yolov8n.engine",
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "imgsz":          224,
    "mjpeg_fps":      20,
    "host":           "0.0.0.0",
    "port":           8000,
    "jetson_id":      "jetson-nano-01",
    "device_name":    "jetson-nano-01",
    "external_api":   "http://10.0.11.153:8080/api/v1/raw_data",

    # TensorRT engine inference is GPU-only. If TensorRT/PyCUDA cannot load, startup stops.
    "gpu_required":   True,

    # Capture stability controls for Jetson Nano.
    # If GStreamer probing crashes or fails, keep this False and use OpenCV/FFmpeg TCP.
    "enable_gstreamer": False,
    # Start capture threads one by one, not all at the same millisecond.
    "capture_startup_stagger_sec": 1.5,
    # For debugging, set to ["cam1"] first. Use None to enable all cameras.
    "active_cameras": None,

    "cameras": {
        "cam1": "rtsp://10.0.11.153:8554/mock1",
        "cam2": "rtsp://10.0.11.153:8554/mock5",
        "cam3": "rtsp://10.0.11.153:8554/mock9",
        "cam4": "rtsp://10.0.11.153:8554/cctv08",
        "cam5": "rtsp://10.0.11.153:8554/cctv09",
    },
}

# ──────────────────────────── COCO class names ────────────────────────────────
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multicam")

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

# ─────────────────────────── TensorRT model (Jetson GPU) ─────────────────────
class HostDeviceMem:
    """Pair of page-locked CPU memory and GPU device memory for one binding."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class YOLOModel:
    """
    TensorRT YOLO inference wrapper for a pre-built .engine file.

    Important:
      - The .engine must be built on the same Jetson / TensorRT / CUDA stack.
      - This class does NOT use ONNX Runtime.
      - Inference runs on the Jetson GPU through TensorRT.
      - Preprocessing, NMS, drawing, and JPEG encoding are still CPU-side.
    """
    def __init__(self, model_path: str):
        if TRT_IMPORT_ERROR is not None or trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT/PyCUDA import failed. This server now requires TensorRT GPU inference.\n"
                f"Import error: {TRT_IMPORT_ERROR}\n\n"
                "Try checking these on Jetson:\n"
                "  python -c 'import tensorrt as trt; print(trt.__version__)'\n"
                "  python -c 'import pycuda.driver as cuda; print(\"pycuda ok\")'\n"
                "If TensorRT works in system Python but not venv, create the venv with:\n"
                "  python3 -m venv --system-site-packages venv\n"
            )

        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.lock = threading.Lock()

        # PyCUDA contexts are thread-local. This server runs inference from
        # camera threads, so create one explicit CUDA context and push/pop it
        # around every TensorRT call. This avoids crashes such as:
        #   free(): double free detected in tcache 2
        cuda.init()
        self.cuda_device = cuda.Device(0)
        self.cuda_ctx = self.cuda_device.make_context()
        self.stream = cuda.Stream()

        log.info("Created explicit CUDA context on device 0")
        log.info(f"Loading TensorRT engine: {model_path}")
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {model_path}. "
                "The .engine may have been built for a different Jetson/TensorRT/CUDA version."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.num_bindings = int(self.engine.num_bindings)
        self.input_indices = []
        self.output_indices = []

        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_indices.append(i)
            else:
                self.output_indices.append(i)

        if len(self.input_indices) != 1:
            raise RuntimeError(f"Expected exactly 1 TensorRT input, found {len(self.input_indices)}")
        if len(self.output_indices) < 1:
            raise RuntimeError("Expected at least 1 TensorRT output.")

        self.input_idx = self.input_indices[0]
        self.output_idx = self.output_indices[0]
        self.input_name = self.engine.get_binding_name(self.input_idx)

        raw_input_shape = tuple(int(x) for x in self.engine.get_binding_shape(self.input_idx))

        # Handle explicit-batch dynamic engines such as [-1, 3, H, W].
        if any(dim < 0 for dim in raw_input_shape):
            self.imgsz = int(CONFIG["imgsz"])
            self.input_shape = (1, 3, self.imgsz, self.imgsz)
            self.context.set_binding_shape(self.input_idx, self.input_shape)
        else:
            self.input_shape = raw_input_shape
            # Supports NCHW. For YOLO exports this is normally [1, 3, imgsz, imgsz].
            if len(self.input_shape) != 4:
                raise RuntimeError(f"Expected NCHW input shape, got {self.input_shape}")
            self.imgsz = int(self.input_shape[2])

        self.bindings = [None] * self.num_bindings
        self.host_device = {}
        self.binding_shapes = {}

        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(int(x) for x in self.context.get_binding_shape(i))
            if any(dim < 0 for dim in shape):
                shape = tuple(int(x) for x in self.engine.get_binding_shape(i))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Binding {name} still has dynamic shape {shape}. Rebuild engine with fixed shape or set profile.")

            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[i] = int(device_mem)
            self.host_device[i] = HostDeviceMem(host_mem, device_mem)
            self.binding_shapes[i] = shape

            role = "input" if self.engine.binding_is_input(i) else "output"
            log.info(f"TensorRT binding {i}: {role} name={name} shape={shape} dtype={dtype}")

        out_shape = self.binding_shapes[self.output_idx]
        self.output_shape = out_shape

        # Detect YOLOv5 [1,N,85] vs YOLOv8 [1,84,N].
        self.is_v5 = (len(out_shape) == 3 and out_shape[2] in (85, 80 + 5))

        self._warmup()

        # Pop the context created in __init__. It will be pushed again inside
        # _execute() from whichever camera thread is running inference.
        self.cuda_ctx.pop()

        log.info(
            f"TensorRT engine loaded on Jetson GPU: {model_path} | "
            f"input={self.input_shape} | output={self.output_shape} | imgsz={self.imgsz} | v5={self.is_v5}"
        )

    def _execute(self, inp):
        # The CUDA context must be current in the thread that touches CUDA.
        self.cuda_ctx.push()
        try:
            inp = np.ascontiguousarray(inp.astype(np.float32, copy=False))
            input_mem = self.host_device[self.input_idx]
            if inp.size != input_mem.host.size:
                raise RuntimeError("Input size mismatch: got {}, engine expects {}".format(inp.shape, self.input_shape))

            np.copyto(input_mem.host, inp.ravel())
            cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)

            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 failed.")

            outputs = []
            for out_idx in self.output_indices:
                out_mem = self.host_device[out_idx]
                cuda.memcpy_dtoh_async(out_mem.host, out_mem.device, self.stream)
                outputs.append((out_idx, out_mem.host))

            self.stream.synchronize()

            reshaped = []
            for out_idx, host in outputs:
                shape = tuple(int(x) for x in self.context.get_binding_shape(out_idx))
                if any(dim < 0 for dim in shape):
                    shape = self.binding_shapes[out_idx]
                reshaped.append(np.array(host).reshape(shape).copy())
            return reshaped
        finally:
            self.cuda_ctx.pop()

    def _warmup(self):
        dummy = np.zeros(self.input_shape, dtype=np.float32)
        with self.lock:
            _ = self._execute(dummy)
        log.info("TensorRT warmup OK")

    def infer_raw(self, inp: np.ndarray):
        """Run TensorRT on pre-built NCHW float32 input. Returns raw output list."""
        with self.lock:
            return self._execute(inp)

    def decode_output(self, out: np.ndarray, orig_hw: tuple) -> list:
        """Decode raw TensorRT output for ONE image into detection dicts."""
        h0, w0 = orig_hw
        if self.is_v5:
            preds = out    # [N, 85]
            mask  = preds[:, 4] * preds[:, 5:].max(axis=1) > CONFIG["conf_threshold"]
            preds = preds[mask]
            if len(preds) == 0:
                return []
            boxes     = preds[:, :4]
            obj       = preds[:, 4]
            cls_probs = preds[:, 5:]
            class_ids = cls_probs.argmax(axis=1)
            scores    = obj * cls_probs[np.arange(len(preds)), class_ids]
        else:
            if len(out.shape) == 3 and out.shape[1] < out.shape[2]:
                preds = out[0].T if out.ndim == 3 else out.T
            else:
                preds = out[0] if out.ndim == 3 else out
            scores_all = preds[:, 4:]
            class_ids  = scores_all.argmax(axis=1)
            scores     = scores_all[np.arange(len(preds)), class_ids]
            mask       = scores > CONFIG["conf_threshold"]
            preds, class_ids, scores = preds[mask], class_ids[mask], scores[mask]
            if len(preds) == 0:
                return []
            boxes = preds[:, :4]

        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * sx
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * sy

        keep = cpu_nms(np.stack([x1,y1,x2,y2], axis=1), scores, CONFIG["iou_threshold"])
        results = []
        for i in keep:
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                "confidence": round(float(scores[i]), 4),
                "bbox_xyxy":  [round(float(x1[i]),1), round(float(y1[i]),1),
                               round(float(x2[i]),1), round(float(y2[i]),1)],
            })
        return results

    def infer_batch(self, batch_chw: np.ndarray, orig_shapes: list) -> list:
        """
        Run TensorRT on a batched [N,3,H,W] float32 tensor.
        orig_shapes: list of (h, w) tuples for each image (for bbox rescaling).
        Returns list of N detection-lists.
        Engine MUST be built with max_batch >= N.
        """
        n = batch_chw.shape[0]
        batch_chw = np.ascontiguousarray(batch_chw.astype(np.float32, copy=False))

        self.cuda_ctx.push()
        try:
            # For dynamic-batch engines, update binding shape.
            raw_in_shape = tuple(int(x) for x in self.engine.get_binding_shape(self.input_idx))
            if any(dim < 0 for dim in raw_in_shape):
                self.context.set_binding_shape(self.input_idx, (n, 3, self.imgsz, self.imgsz))

            out_shape = tuple(int(x) for x in self.context.get_binding_shape(self.output_idx))
            out_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))
            out_size  = int(np.prod(out_shape))

            inp_mem = self.host_device[self.input_idx]
            np.copyto(inp_mem.host[:batch_chw.size], batch_chw.ravel())
            cuda.memcpy_htod_async(inp_mem.device, inp_mem.host, self.stream)

            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 (batch) failed.")

            out_host = cuda.pagelocked_empty(out_size, out_dtype)
            cuda.memcpy_dtoh_async(out_host, self.host_device[self.output_idx].device, self.stream)
            self.stream.synchronize()

            raw = out_host.reshape(out_shape)   # [N, 84, anchors] or [N, anchors, 85]
        finally:
            self.cuda_ctx.pop()

        results = []
        for i in range(n):
            results.append(self.decode_output(raw[i], orig_shapes[i]))
        return results
        h0, w0 = bgr_frame.shape[:2]
        img = cv2.resize(bgr_frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = img.transpose(2, 0, 1)[np.newaxis]  # 1,3,H,W

        with self.lock:
            out = self._execute(inp)[0]

        if self.is_v5:
            preds = out[0]  # [N, 85]
            mask = preds[:, 4] * preds[:, 5:].max(axis=1) > CONFIG["conf_threshold"]
            preds = preds[mask]
            if len(preds) == 0:
                return []
            boxes = preds[:, :4]
            obj   = preds[:, 4]
            cls_probs = preds[:, 5:]
            class_ids = cls_probs.argmax(axis=1)
            scores = obj * cls_probs[np.arange(len(preds)), class_ids]
        else:
            # YOLOv8 export is usually [1, 84, N]. Some engines may return [1, N, 84].
            if len(out.shape) == 3 and out.shape[1] < out.shape[2]:
                preds = out[0].T  # [N, 84]
            elif len(out.shape) == 3:
                preds = out[0]    # [N, 84]
            else:
                raise RuntimeError(f"Unsupported YOLO TensorRT output shape: {out.shape}")

            scores_all = preds[:, 4:]
            class_ids  = scores_all.argmax(axis=1)
            scores     = scores_all[np.arange(len(preds)), class_ids]
            mask = scores > CONFIG["conf_threshold"]
            preds, class_ids, scores = preds[mask], class_ids[mask], scores[mask]
            if len(preds) == 0:
                return []
            boxes = preds[:, :4]

        # xywh -> xyxy in original coords
        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * sx
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * sy

        keep = cpu_nms(np.stack([x1,y1,x2,y2], axis=1), scores, CONFIG["iou_threshold"])
        results = []
        for i in keep:
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                "confidence": round(float(scores[i]), 4),
                "bbox_xyxy":  [round(float(x1[i]),1), round(float(y1[i]),1),
                               round(float(x2[i]),1), round(float(y2[i]),1)],
            })
        return results


def cpu_nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        ix1 = np.maximum(boxes[i,0], boxes[rest,0])
        iy1 = np.maximum(boxes[i,1], boxes[rest,1])
        ix2 = np.minimum(boxes[i,2], boxes[rest,2])
        iy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


# ─────────────────────────── Draw helpers ─────────────────────────────────────
COLORS = [(0,255,0),(255,100,0),(0,100,255),(255,255,0),(0,255,255),
          (255,0,255),(128,255,0),(0,128,255),(255,128,0),(128,0,255)]

def draw_frame(frame, detections, fps, cam_id):
    img = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["bbox_xyxy"]]
        cid = d["class_id"] % len(COLORS)
        color = COLORS[cid]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    # FPS + cam id overlay
    overlay = f"[{cam_id}] {fps:.1f} FPS"
    cv2.putText(img, overlay, (11,31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, overlay, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
    return img


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


# ─────────────────────────── SSE broadcast ────────────────────────────────────
def broadcast_sse(cam_id, record):
    data = json.dumps(record)
    with sse_sub_lock:
        # per-camera subscribers
        for q in per_cam_sse_subscribers[cam_id]:
            try: q.put_nowait(data)
            except: pass
        # global subscribers
        for q in global_sse_subscribers:
            try: q.put_nowait(data)
            except: pass


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


def _postprocess_one(cam_id, frame, detections, fps, frame_idx, latency_ms):
    """
    CPU post-processing: draw bboxes, encode JPEG, update shared state, push SSE/log.
    Runs in a thread pool so the GPU inference loop never waits for it.
    """
    drawn = draw_frame(frame, detections, fps, cam_id)
    _, jpg = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg_bytes = jpg.tobytes()

    state = cam_state[cam_id]
    with state["lock"]:
        state["frame_jpg"]   = jpg_bytes
        state["detections"]  = detections
        state["fps"]         = fps
        state["frame_count"] = frame_idx

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "frame":          frame_idx,
        "timestamp":      ts,
        "camera_id":      cam_id,
        "latency_ms":     latency_ms,
        "fps":            round(fps, 1),
        "num_detections": len(detections),
        "detections":     detections,
    }
    with detection_log_lock:
        detection_log.append(record)
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)

    broadcast_sse(cam_id, record)
    if detections:
        push_external_async(cam_id, record)


def inference_worker(model):
    """
    Batched TensorRT inference worker.

    Strategy:
      1. Every loop tick, collect the LATEST frame from each camera (skip if unchanged).
      2. Preprocess all available frames on CPU in parallel (resize+normalize via numpy).
      3. Stack into a single batch tensor and run ONE TensorRT call for all cameras.
         → GPU sees a larger workload per call → much higher utilisation.
      4. Dispatch draw+JPEG+SSE for each result into a thread-pool so GPU loop never
         waits for slow CPU work.

    Requires a TensorRT engine built with max_batch_size >= num_cameras.
    Build command (run on the same Jetson):
      /usr/src/tensorrt/bin/trtexec \\
        --onnx=yolov8n.onnx \\
        --saveEngine=yolov8n_b5.engine \\
        --fp16 \\
        --minShapes=images:1x3x{imgsz}x{imgsz} \\
        --optShapes=images:5x3x{imgsz}x{imgsz} \\
        --maxShapes=images:5x3x{imgsz}x{imgsz}

    If you still have a batch=1 engine, the worker gracefully falls back to
    sequential batch-1 calls (same as before) but with async postprocessing.
    """
    import concurrent.futures
    postprocess_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(cam_ids), thread_name_prefix="postproc"
    )

    fps_times      = {cid: collections.deque(maxlen=30) for cid in cam_ids}
    last_processed = {cid: -1 for cid in cam_ids}

    imgsz = model.imgsz

    # Detect whether the engine supports batch > 1.
    # engine.max_batch_size is set at build time.
    try:
        engine_max_batch = int(model.engine.max_batch_size)
    except Exception:
        engine_max_batch = 1
    batch_capable = engine_max_batch >= len(cam_ids)

    if batch_capable:
        log.info(f"Inference worker: BATCH MODE (max_batch={engine_max_batch}, cameras={len(cam_ids)})")
    else:
        log.info(
            f"Inference worker: SEQUENTIAL fallback (engine max_batch={engine_max_batch}, "
            f"need {len(cam_ids)}). Rebuild engine with --maxShapes=images:{len(cam_ids)}x3x{imgsz}x{imgsz} for full GPU utilisation."
        )

    def preprocess(frame):
        """Resize + normalize a single BGR frame → float32 CHW."""
        img = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        return img.transpose(2, 0, 1)   # CHW

    while True:
        # ── 1. Collect available frames ──────────────────────────────────────
        ready_cids, ready_frames, ready_idxs = [], [], []
        for cid in cam_ids:
            slot = latest_frame_state[cid]
            with slot["lock"]:
                frame    = None if slot["frame"] is None else slot["frame"]
                fidx     = slot["frame_count"]
            if frame is None or fidx == last_processed[cid]:
                continue
            ready_cids.append(cid)
            ready_frames.append(frame)
            ready_idxs.append(fidx)

        if not ready_cids:
            time.sleep(0.001)
            continue

        t_infer_start = time.time()

        # ── 2. Preprocess all frames in parallel on CPU ───────────────────────
        # Each frame is independent → use thread pool for speed on multi-core CPU.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready_cids)) as pp:
            chw_list = list(pp.map(preprocess, ready_frames))

        # ── 3. TensorRT inference ─────────────────────────────────────────────
        if batch_capable and len(ready_cids) > 1:
            # Stack into [N, 3, H, W] and call GPU once.
            batch_tensor = np.stack(chw_list, axis=0)   # [N, C, H, W]
            try:
                orig_shapes = [f.shape[:2] for f in ready_frames]
                batch_out = model.infer_batch(batch_tensor, orig_shapes)
            except Exception as e:
                log.exception(f"Batch TensorRT inference failed: {e}")
                batch_out = [[] for _ in ready_cids]
        else:
            # Sequential batch=1 calls (fallback or single frame).
            batch_out = []
            for cid, chw in zip(ready_cids, chw_list):
                inp = chw[np.newaxis]   # [1, C, H, W]
                try:
                    outs = model.infer_raw(inp)
                    dets = model.decode_output(outs[0], ready_frames[ready_cids.index(cid)].shape[:2])
                except Exception as e:
                    log.exception(f"[{cid}] TensorRT inference failed: {e}")
                    dets = []
                batch_out.append(dets)

        t_infer_end = time.time()
        total_latency_ms = round((t_infer_end - t_infer_start) * 1000, 1)
        per_frame_ms = round(total_latency_ms / max(len(ready_cids), 1), 1)

        # ── 4. Post-process each result asynchronously ───────────────────────
        for cid, frame, fidx, detections in zip(ready_cids, ready_frames, ready_idxs, batch_out):
            last_processed[cid] = fidx

            times = fps_times[cid]
            times.append(time.time())
            fps = (len(times)-1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0

            # Fire-and-forget: draw + JPEG encode + SSE push in background.
            postprocess_pool.submit(
                _postprocess_one, cid, frame, detections, fps, fidx, per_frame_ms
            )


# ─────────────────────────── FastAPI ──────────────────────────────────────────
app = FastAPI(title="Multi-Cam YOLO TensorRT", version="2.0")
MJPEG_DELAY = 1.0 / CONFIG["mjpeg_fps"]


# ── MJPEG generator ────────────────────────────────────────────────────────────
def mjpeg_gen(cam_id):
    state = cam_state[cam_id]
    while True:
        with state["lock"]:
            jpg = state["frame_jpg"]
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(MJPEG_DELAY)


@app.get("/cam{cam_num}/video")
def cam_video(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return Response("Camera not found", 404)
    return StreamingResponse(mjpeg_gen(cid),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ── SSE per camera ─────────────────────────────────────────────────────────────
async def cam_sse_gen(cam_id):
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    with sse_sub_lock:
        per_cam_sse_subscribers[cam_id].append(q)
    try:
        yield "retry: 3000\n\n"
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=25)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    finally:
        with sse_sub_lock:
            per_cam_sse_subscribers[cam_id].remove(q)


@app.get("/cam{cam_num}/stream")
async def cam_stream(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return Response("Camera not found", 404)
    return StreamingResponse(cam_sse_gen(cid), media_type="text/event-stream")


# ── SSE global (all cameras) ───────────────────────────────────────────────────
async def global_sse_gen():
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    with sse_sub_lock:
        global_sse_subscribers.append(q)
    try:
        yield "retry: 3000\n\n"
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=25)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    finally:
        with sse_sub_lock:
            global_sse_subscribers.remove(q)


@app.get("/log/stream")
async def log_stream():
    return StreamingResponse(global_sse_gen(), media_type="text/event-stream")


# ── Detections query API ───────────────────────────────────────────────────────
@app.get("/detections")
def get_detections(
    limit: int = Query(50, ge=1, le=1000),
    class_name: Optional[str] = None,
    camera_id: Optional[str] = None,
):
    with detection_log_lock:
        results = list(detection_log)
    if class_name:
        results = [r for r in results
                   if any(d["class_name"] == class_name for d in r["detections"])]
    if camera_id:
        results = [r for r in results if r["camera_id"] == camera_id]
    return {"total": len(results), "records": results[-limit:][::-1]}


@app.post("/detections")
async def post_detection(record):
    with detection_log_lock:
        detection_log.append(record)
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)
    cam_id = record.get("camera_id", "external")
    if cam_id in per_cam_sse_subscribers:
        broadcast_sse(cam_id, record)
    return {"ok": True}


# ── loglive redirect ──────────────────────────────────────────────────────────
@app.get("/cam{cam_num}/loglive")
def cam_loglive_redirect(cam_num: int):
    """All /camN/loglive redirect to central /log/live"""
    return RedirectResponse(url="/log/live")


# ── HTML dashboards ────────────────────────────────────────────────────────────
def cam_live_html(cam_id):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{cam_id.upper()} Live</title>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:monospace;}}
  h2{{margin:8px 12px;color:#0f0;}}
  .nav{{display:flex;gap:6px;padding:6px 12px;flex-wrap:wrap;}}
  .nav a{{color:#0af;text-decoration:none;padding:4px 8px;border:1px solid #0af;border-radius:4px;font-size:13px;}}
  .nav a:hover{{background:#0af;color:#000;}}
  .main{{display:flex;gap:12px;padding:12px;}}
  #video{{max-width:720px;width:100%;border:2px solid #0f0;border-radius:4px;}}
  #log{{flex:1;max-height:500px;overflow-y:auto;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:8px;font-size:12px;}}
  .row{{padding:3px 0;border-bottom:1px solid #222;animation:flash 0.6s ease;}}
  @keyframes flash{{from{{background:#0f02;}}to{{background:transparent;}}}}
  #stats{{display:flex;gap:16px;padding:4px 12px;font-size:13px;background:#1a1a1a;border-bottom:1px solid #333;}}
  .stat{{color:#aaa;}} .stat span{{color:#0f0;font-weight:bold;}}
</style>
</head>
<body>
<h2>📷 {cam_id.upper()} — Live Detection</h2>
<div class="nav">
  {"".join(f'<a href="/cam{i}/live">cam{i}</a>' for i in range(1,6))}
  <a href="/log/live" style="border-color:#f80;color:#f80;">📋 Central Log</a>
</div>
<div id="stats">
  <div class="stat">FPS: <span id="fps">—</span></div>
  <div class="stat">Latency: <span id="lat">—</span> ms</div>
  <div class="stat">Detections: <span id="dets">—</span></div>
  <div class="stat">Frame: <span id="frm">—</span></div>
</div>
<div class="main">
  <img id="video" src="/{cam_id}/video">
  <div id="log"><em>Waiting for detections...</em></div>
</div>
<script>
const src = new EventSource('/{cam_id}/stream');
const logEl = document.getElementById('log');
src.onmessage = e => {{
  const r = JSON.parse(e.data);
  document.getElementById('fps').textContent = r.fps?.toFixed(1) ?? '—';
  document.getElementById('lat').textContent = r.latency_ms ?? '—';
  document.getElementById('dets').textContent = r.num_detections ?? 0;
  document.getElementById('frm').textContent = r.frame ?? '—';
  if (r.num_detections > 0) {{
    const div = document.createElement('div');
    div.className = 'row';
    const names = r.detections.map(d => `${{d.class_name}}(${{(d.confidence*100).toFixed(0)}}%)`).join(', ');
    div.textContent = `[${{r.timestamp.slice(11,19)}}] frame ${{r.frame}}: ${{names}}`;
    if (logEl.firstChild?.tagName === 'EM') logEl.innerHTML = '';
    logEl.prepend(div);
    if (logEl.children.length > 200) logEl.lastChild.remove();
  }}
}};
src.onerror = () => setTimeout(() => location.reload(), 3000);
</script>
</body></html>"""


def log_live_html():
    cam_links = "".join(
        f'<a href="/cam{i}/live">cam{i}</a> ' for i in range(1, 6)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Central Detection Log</title>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:monospace;}}
  h2{{margin:8px 12px;color:#f80;}}
  .nav{{display:flex;gap:6px;padding:6px 12px;flex-wrap:wrap;}}
  .nav a{{color:#0af;text-decoration:none;padding:4px 8px;border:1px solid #0af;border-radius:4px;font-size:13px;}}
  .nav a:hover{{background:#0af;color:#000;}}
  #stats{{display:flex;gap:16px;padding:6px 12px;font-size:13px;background:#1a1a1a;border-bottom:1px solid #333;flex-wrap:wrap;}}
  .stat{{color:#aaa;}} .stat span{{color:#f80;font-weight:bold;}}
  #log{{padding:10px 12px;max-height:calc(100vh - 160px);overflow-y:auto;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th{{background:#222;color:#aaa;padding:5px 8px;text-align:left;position:sticky;top:0;}}
  td{{padding:4px 8px;border-bottom:1px solid #1e1e1e;}}
  tr.new{{animation:flash 0.6s ease;}}
  @keyframes flash{{from{{background:#f801;}}to{{background:transparent;}}}}
  .cam1{{color:#0f0;}} .cam2{{color:#0af;}} .cam3{{color:#f80;}}
  .cam4{{color:#f0f;}} .cam5{{color:#ff0;}}
  #chart-wrap{{padding:0 12px 8px;}}
  canvas{{background:#1a1a1a;border-radius:4px;}}
</style>
</head>
<body>
<h2>📋 Central Detection Log — All Cameras</h2>
<div class="nav">
  {cam_links}
  <a href="/log/live" style="border-color:#f80;color:#f80;">📋 Central Log</a>
  <a href="/detections" style="border-color:#aaa;color:#aaa;">🔌 API</a>
</div>
<div id="stats">
  {"".join(f'<div class="stat">cam{i}: <span id="cnt{i}">0</span></div>' for i in range(1,6))}
  <div class="stat">Total rows: <span id="total">0</span></div>
</div>
<div id="chart-wrap">
  <canvas id="chart" height="60"></canvas>
</div>
<div id="log">
  <table>
    <thead><tr>
      <th>Time</th><th>Camera</th><th>Frame</th>
      <th>Detections</th><th>Latency</th><th>Objects</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<script>
const camColors = {{cam1:'#0f0',cam2:'#0af',cam3:'#f80',cam4:'#f0f',cam5:'#ff0'}};
const counts = {{cam1:0,cam2:0,cam3:0,cam4:0,cam5:0}};
const classCounts = {{}};
let total = 0;

const src = new EventSource('/log/stream');
src.onmessage = e => {{
  const r = JSON.parse(e.data);
  const cam = r.camera_id || 'unknown';
  if (counts[cam] !== undefined) counts[cam]++;
  total++;
  document.getElementById('total').textContent = total;
  for (let i=1;i<=5;i++) {{
    const el = document.getElementById('cnt'+i);
    if (el) el.textContent = counts['cam'+i] || 0;
  }}
  r.detections?.forEach(d => {{
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  }});
  drawChart();
  if (r.num_detections === 0) return;
  const tbody = document.getElementById('tbody');
  const tr = document.createElement('tr');
  tr.className = 'new';
  const names = r.detections.map(d=>`${{d.class_name}}(${{(d.confidence*100).toFixed(0)}}%)`).join(', ');
  const camColor = camColors[cam] || '#eee';
  tr.innerHTML = `
    <td>${{r.timestamp?.slice(11,19) ?? ''}}</td>
    <td style="color:${{camColor}};font-weight:bold">${{cam}}</td>
    <td>${{r.frame ?? ''}}</td>
    <td>${{r.num_detections}}</td>
    <td>${{r.latency_ms}} ms</td>
    <td>${{names}}</td>`;
  tbody.prepend(tr);
  if (tbody.children.length > 500) tbody.lastChild.remove();
}};
src.onerror = () => setTimeout(() => location.reload(), 3000);

function drawChart() {{
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth - 24;
  canvas.width = W; canvas.height = 80;
  ctx.clearRect(0,0,W,80);
  const entries = Object.entries(classCounts).sort((a,b)=>b[1]-a[1]).slice(0,10);
  if (!entries.length) return;
  const max = entries[0][1];
  const bw = Math.max(20, Math.floor((W - 20) / entries.length) - 4);
  entries.forEach(([cls, cnt], i) => {{
    const bh = Math.max(4, Math.floor((cnt / max) * 60));
    const x = 10 + i * (bw + 4);
    const y = 70 - bh;
    ctx.fillStyle = '#0f0';
    ctx.fillRect(x, y, bw, bh);
    ctx.fillStyle = '#aaa';
    ctx.font = '9px monospace';
    ctx.fillText(cls.slice(0,7), x, 80);
  }});
}}
</script>
</body></html>"""


@app.get("/cam{cam_num}/live", response_class=HTMLResponse)
def cam_live(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return HTMLResponse("Camera not found", 404)
    return HTMLResponse(cam_live_html(cid))


@app.get("/log/live", response_class=HTMLResponse)
def log_live():
    return HTMLResponse(log_live_html())


@app.get("/", response_class=HTMLResponse)
def root():
    links = "".join(
        f'<li><a href="/cam{i}/live">/cam{i}/live</a> — Camera {i} dashboard</li>'
        for i in range(1, 6)
    )
    return HTMLResponse(f"""<html><body style="background:#111;color:#eee;font-family:monospace;padding:24px;">
<h2 style="color:#0f0">Multi-Cam YOLO TensorRT Server</h2>
<ul>{links}
<li><a href="/log/live" style="color:#f80">/log/live</a> — Central log (all cameras)</li>
<li><a href="/detections">/detections</a> — JSON API</li>
<li><a href="/docs">/docs</a> — Swagger UI</li>
</ul></body></html>""")


# ─────────────────────────── Startup ──────────────────────────────────────────
def start_pipelines(model):
    # Start RTSP capture threads first. They only read frames and update latest_frame_state.
    for cam_id, source in CONFIG["cameras"].items():
        t = threading.Thread(
            target=capture_loop,
            args=(cam_id, source),
            daemon=True,
            name=f"capture-{cam_id}",
        )
        t.start()
        log.info(f"Started capture thread: {cam_id}")
        time.sleep(float(CONFIG.get("capture_startup_stagger_sec", 0.0)))

    # Start exactly one TensorRT inference thread. CUDA/TensorRT is used only here.
    t = threading.Thread(
        target=inference_worker,
        args=(model,),
        daemon=True,
        name="tensorrt-inference-worker",
    )
    t.start()
    log.info("Started TensorRT inference worker")


if __name__ == "__main__":
    log.info("Loading YOLO model...")
    model = YOLOModel(CONFIG["model_path"])

    log.info("Starting camera pipeline threads...")
    start_pipelines(model)

    log.info(f"Starting FastAPI server on {CONFIG['host']}:{CONFIG['port']}")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning")