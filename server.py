# -*- coding: utf-8 -*-
"""
Multi-Camera YOLO TensorRT Jetson Pipeline
5 RTSP streams -> single YOLO model -> per-camera MJPEG + SSE + log
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import uvicorn

# ─────────────────────────── CONFIG (edit only here) ──────────────────────────
CONFIG = {
    "model_path":     "yolov8n_fp16.engine",
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "imgsz":          224,
    "mjpeg_fps":      20,
    "use_gstreamer":  True,
    "rtsp_latency_ms": 100,
    "host":           "0.0.0.0",
    "port":           8080,
    "jetson_id":      "jetson-nano-01",
    "device_name":    "jetson-nano-01",
    "external_api":   "http://10.0.11.153:8080/api/v1/raw_data",
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
cam_ids = list(CONFIG["cameras"].keys())

cam_state: Dict[str, dict] = {
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

# Central detection log (all cameras)
detection_log: List[dict] = []
detection_log_lock = threading.Lock()
MAX_LOG = 5000

# SSE queues: per-camera subscribers + global subscribers
global_sse_subscribers: List[asyncio.Queue] = []
per_cam_sse_subscribers: Dict[str, List[asyncio.Queue]] = {cid: [] for cid in cam_ids}
sse_sub_lock = threading.Lock()

# ───────────────────────── TensorRT model (shared, thread-safe) ───────────────
class YOLOModel:
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file
        if not model_file.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {model_file}")

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT engine inference requires Python packages 'tensorrt' "
                "and 'pycuda' on the Jetson runtime."
            ) from exc

        self.trt = trt
        self.cuda = cuda
        self.lock = threading.Lock()
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
        try:
            runtime = trt.Runtime(self.trt_logger)
            with model_file.open("rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_file}")

            self.context = self.engine.create_execution_context()
            self.uses_tensor_api = hasattr(self.engine, "num_io_tensors")
            self.bindings = self._inspect_bindings()

            inputs = [b for b in self.bindings if b["is_input"]]
            if len(inputs) != 1:
                raise RuntimeError(f"Expected one engine input, found {len(inputs)}")
            self.input_binding = inputs[0]
            self.output_bindings = [b for b in self.bindings if not b["is_input"]]
            if not self.output_bindings:
                raise RuntimeError("TensorRT engine has no output bindings")

            self.input_shape = self._resolve_input_shape(self.input_binding["shape"])
            self._set_input_shape(self.input_binding["name"], self.input_shape)
            self._derive_input_layout()
            self._allocate_buffers()
        finally:
            self.cuda_context.pop()

        output_desc = ", ".join(
            f"{b['name']}:{tuple(b['shape'])}/{np.dtype(b['dtype']).name}"
            for b in self.output_bindings
        )
        log.info(
            "TensorRT engine loaded: %s | input=%s:%s/%s | outputs=%s",
            model_path,
            self.input_binding["name"],
            self.input_shape,
            np.dtype(self.input_binding["dtype"]).name,
            output_desc,
        )

    def _inspect_bindings(self) -> List[dict]:
        bindings = []
        if self.uses_tensor_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                bindings.append({
                    "index": i,
                    "name": name,
                    "is_input": mode == self.trt.TensorIOMode.INPUT,
                    "shape": tuple(self.engine.get_tensor_shape(name)),
                    "dtype": np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name))),
                })
        else:
            for i in range(self.engine.num_bindings):
                bindings.append({
                    "index": i,
                    "name": self.engine.get_binding_name(i),
                    "is_input": self.engine.binding_is_input(i),
                    "shape": tuple(self.engine.get_binding_shape(i)),
                    "dtype": np.dtype(self.trt.nptype(self.engine.get_binding_dtype(i))),
                })
        return bindings

    def _resolve_input_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(shape) != 4:
            raise RuntimeError(f"Expected 4D YOLO input, got {shape}")

        resolved = [int(d) for d in shape]
        if resolved[0] < 0:
            resolved[0] = 1

        # Prefer the engine's static dimensions. CONFIG["imgsz"] only fills
        # dynamic dimensions when the engine was built with an optimization profile.
        if resolved[1] == 3 or resolved[1] < 0:
            if resolved[1] < 0:
                resolved[1] = 3
            if resolved[2] < 0:
                resolved[2] = int(CONFIG["imgsz"])
            if resolved[3] < 0:
                resolved[3] = int(CONFIG["imgsz"])
        elif resolved[3] == 3 or resolved[3] < 0:
            if resolved[1] < 0:
                resolved[1] = int(CONFIG["imgsz"])
            if resolved[2] < 0:
                resolved[2] = int(CONFIG["imgsz"])
            if resolved[3] < 0:
                resolved[3] = 3
        else:
            raise RuntimeError(f"Cannot infer YOLO input layout from shape {shape}")

        return tuple(resolved)

    def _set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        if self.uses_tensor_api and hasattr(self.context, "set_input_shape"):
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"TensorRT rejected input shape {shape} for {name}")
        elif any(d < 0 for d in self.input_binding["shape"]):
            if not self.context.set_binding_shape(self.input_binding["index"], shape):
                raise RuntimeError(f"TensorRT rejected input shape {shape} for {name}")

    def _derive_input_layout(self) -> None:
        shape = self.input_shape
        if shape[1] == 3:
            self.input_layout = "NCHW"
            self.input_h = int(shape[2])
            self.input_w = int(shape[3])
        elif shape[3] == 3:
            self.input_layout = "NHWC"
            self.input_h = int(shape[1])
            self.input_w = int(shape[2])
        else:
            raise RuntimeError(f"Unsupported YOLO input shape: {shape}")
        self.imgsz = self.input_h if self.input_h == self.input_w else (self.input_h, self.input_w)

    def _binding_runtime_shape(self, binding: dict) -> Tuple[int, ...]:
        if self.uses_tensor_api and hasattr(self.context, "get_tensor_shape"):
            shape = tuple(self.context.get_tensor_shape(binding["name"]))
        elif hasattr(self.context, "get_binding_shape"):
            shape = tuple(self.context.get_binding_shape(binding["index"]))
        else:
            shape = binding["shape"]

        if any(int(d) < 0 for d in shape):
            raise RuntimeError(
                f"Dynamic shape for binding {binding['name']} was not resolved: {shape}"
            )
        return tuple(int(d) for d in shape)

    def _allocate_buffers(self) -> None:
        self.stream = self.cuda.Stream()
        self.binding_ptrs = [0] * len(self.bindings)

        for binding in self.bindings:
            shape = self._binding_runtime_shape(binding)
            binding["shape"] = shape
            size = int(np.prod(shape))
            host = self.cuda.pagelocked_empty(size, binding["dtype"])
            device = self.cuda.mem_alloc(host.nbytes)
            binding["host"] = host
            binding["device"] = device
            self.binding_ptrs[binding["index"]] = int(device)

            if self.uses_tensor_api and hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(binding["name"], int(device))

    def _preprocess(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h0, w0 = bgr_frame.shape[:2]
        scale = min(self.input_w / w0, self.input_h / h0)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        if np.issubdtype(self.input_binding["dtype"], np.floating):
            tensor = canvas.astype(np.float32) / 255.0
            tensor = tensor.astype(self.input_binding["dtype"], copy=False)
        else:
            tensor = canvas.astype(self.input_binding["dtype"], copy=False)

        if self.input_layout == "NCHW":
            tensor = tensor.transpose(2, 0, 1)[np.newaxis]
        else:
            tensor = tensor[np.newaxis]
        return np.ascontiguousarray(tensor), scale, (pad_x, pad_y)

    def infer(self, bgr_frame: np.ndarray) -> List[dict]:
        h0, w0 = bgr_frame.shape[:2]
        inp, scale, pad = self._preprocess(bgr_frame)

        with self.lock:
            self.cuda_context.push()
            try:
                np.copyto(self.input_binding["host"], inp.ravel())
                self.cuda.memcpy_htod_async(
                    self.input_binding["device"],
                    self.input_binding["host"],
                    self.stream,
                )

                if self.uses_tensor_api and hasattr(self.context, "execute_async_v3"):
                    ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
                else:
                    ok = self.context.execute_async_v2(
                        bindings=self.binding_ptrs,
                        stream_handle=self.stream.handle,
                    )
                if not ok:
                    raise RuntimeError("TensorRT inference failed")

                for binding in self.output_bindings:
                    self.cuda.memcpy_dtoh_async(
                        binding["host"],
                        binding["device"],
                        self.stream,
                    )
                self.stream.synchronize()
                outputs = [
                    binding["host"].reshape(binding["shape"]).copy()
                    for binding in self.output_bindings
                ]
            finally:
                self.cuda_context.pop()

        return self._postprocess(outputs, h0, w0, scale, pad)

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        orig_h: int,
        orig_w: int,
        scale: float,
        pad: Tuple[int, int],
    ) -> List[dict]:
        multi_output = self._postprocess_multi_output_nms(outputs, orig_h, orig_w, scale, pad)
        if multi_output is not None:
            return multi_output

        out = np.asarray(outputs[0], dtype=np.float32)
        preds = self._as_prediction_matrix(out)
        if preds.size == 0:
            return []

        if preds.shape[1] == 6:
            boxes_xyxy = preds[:, :4]
            scores = preds[:, 4]
            class_ids = preds[:, 5].astype(np.int32)
        elif preds.shape[1] == 85:
            obj = preds[:, 4]
            cls_probs = preds[:, 5:]
            class_ids = cls_probs.argmax(axis=1)
            scores = obj * cls_probs[np.arange(len(preds)), class_ids]
            boxes_xyxy = xywh_to_xyxy(preds[:, :4])
        elif preds.shape[1] >= 84:
            cls_probs = preds[:, 4:]
            class_ids = cls_probs.argmax(axis=1)
            scores = cls_probs[np.arange(len(preds)), class_ids]
            boxes_xyxy = xywh_to_xyxy(preds[:, :4])
        else:
            raise RuntimeError(f"Unsupported YOLO output shape: {out.shape}")

        mask = scores > CONFIG["conf_threshold"]
        if not np.any(mask):
            return []

        boxes_xyxy = boxes_xyxy[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        boxes_xyxy = scale_boxes_from_engine(boxes_xyxy, orig_w, orig_h, scale, pad)

        keep = cpu_nms(boxes_xyxy, scores, CONFIG["iou_threshold"])
        results = []
        for i in keep:
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                "confidence": round(float(scores[i]), 4),
                "bbox_xyxy":  [round(float(v), 1) for v in boxes_xyxy[i]],
            })
        return results

    def _as_prediction_matrix(self, out: np.ndarray) -> np.ndarray:
        out = np.squeeze(out)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        if out.ndim != 2:
            raise RuntimeError(f"Unsupported YOLO output shape: {out.shape}")

        # Ultralytics YOLOv8 TensorRT exports normally return [84, N].
        # Some engines/plugins return [N, 84] or [N, 6].
        if out.shape[0] in (84, 85) and out.shape[1] != out.shape[0]:
            out = out.T
        elif out.shape[0] < out.shape[1] and out.shape[0] > 6:
            out = out.T
        return out

    def _postprocess_multi_output_nms(
        self,
        outputs: List[np.ndarray],
        orig_h: int,
        orig_w: int,
        scale: float,
        pad: Tuple[int, int],
    ) -> Optional[List[dict]]:
        if len(outputs) < 4:
            return None

        arrays = [np.asarray(o, dtype=np.float32) for o in outputs]
        boxes_idx = next(
            (i for i, arr in enumerate(arrays) if np.squeeze(arr).ndim >= 2 and np.squeeze(arr).shape[-1] == 4),
            None,
        )
        if boxes_idx is None:
            return None

        boxes = np.squeeze(arrays[boxes_idx]).reshape(-1, 4)
        count = len(boxes)
        scalar_counts = [
            int(np.ravel(arr)[0])
            for i, arr in enumerate(arrays)
            if i != boxes_idx and np.ravel(arr).size == 1
        ]
        if scalar_counts:
            count = min(count, max(0, scalar_counts[0]))
        if count == 0:
            return []

        candidates = []
        for i, arr in enumerate(arrays):
            if i == boxes_idx or np.ravel(arr).size == 1:
                continue
            flat = np.squeeze(arr).reshape(-1)
            if len(flat) >= count:
                candidates.append((i, flat[:count]))
        if len(candidates) < 2:
            return None

        names = [b["name"].lower() for b in self.output_bindings]
        named_score_idx = next(
            (i for i, _ in candidates if "score" in names[i] or "conf" in names[i]),
            None,
        )
        named_class_idx = next(
            (i for i, _ in candidates if "class" in names[i] or "label" in names[i]),
            None,
        )

        if named_score_idx is not None and named_class_idx is not None:
            scores = next(values for i, values in candidates if i == named_score_idx)
            class_ids = next(values for i, values in candidates if i == named_class_idx).astype(np.int32)
        else:
            score_candidates = [
                item for item in candidates
                if np.nanmin(item[1]) >= 0.0 and np.nanmax(item[1]) <= 1.0
            ]
            score_idx, scores = score_candidates[0] if score_candidates else candidates[0]
            class_candidates = [item for item in candidates if item[0] != score_idx]
            class_ids = class_candidates[0][1].astype(np.int32)

        boxes = boxes[:count]
        scores = scores[:count]
        mask = scores > CONFIG["conf_threshold"]
        if not np.any(mask):
            return []

        boxes = scale_boxes_from_engine(boxes[mask], orig_w, orig_h, scale, pad)
        scores = scores[mask]
        class_ids = class_ids[mask]
        keep = cpu_nms(boxes, scores, CONFIG["iou_threshold"])

        results = []
        for i in keep:
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                "confidence": round(float(scores[i]), 4),
                "bbox_xyxy":  [round(float(v), 1) for v in boxes[i]],
            })
        return results


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=False)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def scale_boxes_from_engine(
    boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad: Tuple[int, int],
) -> np.ndarray:
    pad_x, pad_y = pad
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


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

def draw_frame(frame: np.ndarray, detections: List[dict], fps: float, cam_id: str) -> np.ndarray:
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


# ─────────────────────────── External API push ────────────────────────────────
def push_to_external_api(cam_id: str, record: dict):
    """POST detection record to external API (non-blocking, best-effort)."""
    for det in record.get("detections", []):
        bbox = det.get("bbox_xyxy", [0,0,0,0])
        x1,y1,x2,y2 = bbox
        payload = {
            "device_name": CONFIG["device_name"],
            "data": {
                "timestamp":  record["timestamp"],
                "type":       det["class_name"],
                "color":      "",           # not yet available from model
                "brand":      "",           # not yet available from model
                "x":          round(float(x1), 4),
                "y":          round(float(y1), 4),
                "width":      round(float(x2 - x1), 4),
                "height":     round(float(y2 - y1), 4),
                "camera_id":  cam_id,
                "jetson_id":  CONFIG["jetson_id"],
                "track_id":   str(det.get("track_id", "")),
            }
        }
        try:
            requests.post(CONFIG["external_api"], json=payload, timeout=2)
        except Exception as e:
            log.warning(f"[{cam_id}] External API push failed: {e}")


def push_external_async(cam_id: str, record: dict):
    t = threading.Thread(target=push_to_external_api, args=(cam_id, record), daemon=True)
    t.start()


# ─────────────────────────── SSE broadcast ────────────────────────────────────
def broadcast_sse(cam_id: str, record: dict):
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
def gst_rtsp_pipelines(source: str):
    latency = int(CONFIG.get("rtsp_latency_ms", 100))
    appsink = "appsink drop=1 max-buffers=1 sync=false"

    # JetPack 4/5/6 preferred path. nvv4l2decoder uses Jetson HW decode.
    yield (
        "nvv4l2decoder",
        f"rtspsrc location={source} latency={latency} protocols=tcp ! "
        "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! {appsink}",
    )

    # Older JetPack fallback.
    yield (
        "omxh264dec",
        f"rtspsrc location={source} latency={latency} protocols=tcp ! "
        "rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! "
        "video/x-raw,format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! {appsink}",
    )

    # CPU decode fallback through GStreamer. Still useful when OpenCV's direct
    # RTSP backend is unreliable, but it will not reduce CPU load.
    yield (
        "avdec_h264",
        f"rtspsrc location={source} latency={latency} protocols=tcp ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        f"video/x-raw,format=BGR ! {appsink}",
    )


def open_capture(cam_id: str, source: str) -> cv2.VideoCapture:
    """Try GStreamer first, then fall back to OpenCV's default RTSP backend."""
    if CONFIG.get("use_gstreamer", True) and source.startswith("rtsp://"):
        for name, gst in gst_rtsp_pipelines(source):
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                log.info(f"[{cam_id}] GStreamer opened with {name}")
                return cap
            cap.release()
            log.warning(f"[{cam_id}] GStreamer pipeline failed: {name}")

        log.warning(f"[{cam_id}] All GStreamer pipelines failed, falling back to OpenCV RTSP")

    cap = cv2.VideoCapture(source)
    return cap


def pipeline_loop(cam_id: str, source: str, model: YOLOModel):
    state = cam_state[cam_id]
    fps_times: collections.deque = collections.deque(maxlen=30)
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

            t0 = time.time()
            detections = model.infer(frame)
            latency_ms = round((time.time() - t0) * 1000, 1)

            fps_times.append(time.time())
            if len(fps_times) >= 2:
                fps = (len(fps_times)-1) / (fps_times[-1] - fps_times[0])
            else:
                fps = 0.0

            drawn = draw_frame(frame, detections, fps, cam_id)
            _, jpg = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_bytes = jpg.tobytes()

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

            # Push to external API in background thread (best-effort)
            if detections:
                push_external_async(cam_id, record)

            frame_idx += 1

        cap.release()
        time.sleep(2)


# ─────────────────────────── FastAPI ──────────────────────────────────────────
app = FastAPI(title="Multi-Cam YOLO TensorRT", version="2.0")
MJPEG_DELAY = 1.0 / CONFIG["mjpeg_fps"]


# ── MJPEG generator ────────────────────────────────────────────────────────────
def mjpeg_gen(cam_id: str):
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
async def cam_sse_gen(cam_id: str):
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
async def post_detection(record: dict):
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
def cam_live_html(cam_id: str) -> str:
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


def log_live_html() -> str:
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
def start_pipelines(model: YOLOModel):
    for cam_id, source in CONFIG["cameras"].items():
        t = threading.Thread(
            target=pipeline_loop,
            args=(cam_id, source, model),
            daemon=True,
            name=f"pipeline-{cam_id}",
        )
        t.start()
        log.info(f"Started pipeline thread: {cam_id}")


if __name__ == "__main__":
    log.info("Loading YOLO model...")
    model = YOLOModel(CONFIG["model_path"])

    log.info("Starting camera pipeline threads...")
    start_pipelines(model)

    log.info(f"Starting FastAPI server on {CONFIG['host']}:{CONFIG['port']}")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning")
