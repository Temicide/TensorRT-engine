# -*- coding: utf-8 -*-
"""
Multi-Camera YOLO Jetson Pipeline - safe capture
5 RTSP streams -> detector backend -> per-camera MJPEG + SSE + log
All detection logs POST to: http://10.0.11.153:8080/api/v1/raw_data

Endpoints:
  /cam{1-5}/live        -> MJPEG dashboard per camera
  /cam{1-5}/video       -> MJPEG stream per camera
  /cam{1-5}/stream      -> SSE per camera
  /cam{1-5}/loglive     -> redirect to central log dashboard
  /log/live             -> central log dashboard (all cameras)
  /log/stream           -> SSE for all cameras combined
  /detections           -> query log (?limit=50&class_name=person&camera_id=cam1)

Start from editing config.py
"""
import os
import logging
from core.model_factory import load_detector_model
from config import CONFIG
from core.pipeline import start_pipelines
from routers.cameras import router as cameras_router
from routers.detections import router as detections_router
from routers.logs import router as logs_router

# Make OpenCV/FFmpeg RTSP capture more stable on Jetson Nano.
# These options must be set before cv2 opens any VideoCapture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;500000")

import cv2
# Avoid OpenCV spawning many internal CPU threads on Jetson Nano.
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

log = logging.getLogger("multicam")

# ─────────────────────────── FastAPI ──────────────────────────────────────────
app = FastAPI(title="Multi-Cam YOLO Jetson", version="2.1")

app.include_router(cameras_router)
app.include_router(logs_router)
app.include_router(detections_router)

@app.get("/", response_class=HTMLResponse)
def root():
    links = "".join(
        f'<li><a href="/cam{i}/live">/cam{i}/live</a> — Camera {i} dashboard</li>'
        for i in range(1, 6)
    )
    return HTMLResponse(f"""<html><body style="background:#111;color:#eee;font-family:monospace;padding:24px;">
<h2 style="color:#0f0">Multi-Cam YOLO Jetson Server</h2>
<p>Backend: {CONFIG.get("inference_backend", "tensorrt_engine")}</p>
<ul>{links}
<li><a href="/log/live" style="color:#f80">/log/live</a> — Central log (all cameras)</li>
<li><a href="/detections">/detections</a> — JSON API</li>
<li><a href="/docs">/docs</a> — Swagger UI</li>
</ul></body></html>""")


# ─────────────────────────── Startup ──────────────────────────────────────────

if __name__ == "__main__":
    log.info("Loading YOLO model backend: %s", CONFIG.get("inference_backend", "tensorrt_engine"))
    model = load_detector_model()

    log.info("Starting camera pipeline threads...")
    start_pipelines(model)

    log.info(f"Starting FastAPI server on {CONFIG['host']}:{CONFIG['port']}")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning")
