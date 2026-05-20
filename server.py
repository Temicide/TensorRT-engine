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

import json
import uuid
import asyncio
import os
import threading
import time
import collections
import logging
from datetime import datetime, timezone
from typing import Optional
from static.templates import cam_live_html, log_live_html
from core.tensorrt_engine import YOLOModel
from routers.api_exporter import push_external_async
from utils.utils import draw_frame, COLORS, COCO_NAMES
from config import CONFIG

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
from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import uvicorn

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
    for cam_id in cam_ids:
        source = CONFIG["cameras"][cam_id]
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