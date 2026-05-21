import collections
import logging
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from config import CONFIG
from core.camera_capture import capture_loop
from core.sse import broadcast_sse
from core.state import cam_ids, cam_state, detection_log, detection_log_lock, latest_frame_state, MAX_LOG
from routers.api_exporter import push_external_async
from utils.utils import draw_frame

log = logging.getLogger("multicam")

def inference_worker(model):
    """
    Detector inference worker.

    Strategy:
      1. Every loop tick, collect the LATEST frame from each camera (skip if unchanged).
      2. For TensorRT models, preprocess available frames on CPU in parallel.
      3. If supported, stack into a single batch tensor and run ONE TensorRT call.
         → GPU sees a larger workload per call → much higher utilisation.
      4. For non-TensorRT backends, call model.infer_frame(frame) per camera.
      5. Dispatch draw+JPEG+SSE for each result into a thread-pool so inference never
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

    imgsz = getattr(model, "imgsz", int(CONFIG.get("imgsz", 416)))
    backend_name = getattr(model, "backend_name", "tensorrt_engine")
    has_tensorrt_api = all(
        hasattr(model, attr)
        for attr in ("engine", "infer_batch", "infer_raw", "decode_output")
    )

    # Detect whether the engine supports batch > 1.
    # engine.max_batch_size is set at build time.
    if has_tensorrt_api:
        try:
            engine_max_batch = int(model.engine.max_batch_size)
        except Exception:
            engine_max_batch = 1
    else:
        engine_max_batch = 1
    batch_capable = has_tensorrt_api and engine_max_batch >= len(cam_ids)

    if batch_capable:
        log.info(f"Inference worker: BATCH MODE (max_batch={engine_max_batch}, cameras={len(cam_ids)})")
    elif has_tensorrt_api:
        log.info(
            f"Inference worker: SEQUENTIAL fallback (engine max_batch={engine_max_batch}, "
            f"need {len(cam_ids)}). Rebuild engine with --maxShapes=images:{len(cam_ids)}x3x{imgsz}x{imgsz} for full GPU utilisation."
        )
    else:
        log.info("Inference worker: PER-FRAME MODE backend=%s cameras=%s pool=%s", backend_name, len(cam_ids), getattr(model, "num_instances", 1))
    
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

        if has_tensorrt_api:
            # ── 2. Preprocess all frames in parallel on CPU ───────────────────
            # Each frame is independent → use thread pool for speed on multi-core CPU.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready_cids)) as pp:
                chw_list = list(pp.map(preprocess, ready_frames))

            # ── 3. TensorRT inference ─────────────────────────────────────────
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
        else:
            # Generic non-TensorRT backend (e.g. tkDNN through a bridge).
            if hasattr(model, "infer_frames_parallel"):
                batch_out = model.infer_frames_parallel(ready_cids, ready_frames)
            else:
                batch_out = []
                for cid, frame in zip(ready_cids, ready_frames):
                    try:
                        dets = model.infer_frame(frame)
                    except RuntimeError as e:
                        log.error("[%s] %s inference failed: %s", cid, backend_name, e)
                        dets = []
                    except Exception as e:
                        log.exception("[%s] %s inference failed: %s", cid, backend_name, e)
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

    # Start exactly one inference thread. GPU runtimes are used only here.
    t = threading.Thread(
        target=inference_worker,
        args=(model,),
        daemon=True,
        name="detector-inference-worker",
    )
    t.start()
    log.info("Started detector inference worker")
