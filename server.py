# -*- coding: utf-8 -*-
"""
Multi-Camera YOLO DeepStream Jetson Nano Pipeline
RTSP -> NVIDIA decoder -> nvstreammux -> nvinfer -> nvdsosd -> sink
DeepStream metadata -> per-camera SSE + log + external API

Endpoints:
  /cam{N}/live          -> MJPEG/SSE dashboard per active camera
  /cam{N}/video         -> MJPEG stream if DeepStream appsink output is enabled
  /cam{N}/stream        -> SSE per camera
  /cam{N}/loglive       -> redirect to central log dashboard
  /log/live             -> central log dashboard
  /log/stream           -> SSE for all cameras combined
  /detections           -> query log (?limit=50&class_name=person&camera_id=cam1)

Start from editing config.py.
"""
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from config import CONFIG
from core.pipeline import start_pipelines, stop_pipelines
from core.state import cam_ids
from routers.cameras import router as cameras_router
from routers.detections import router as detections_router
from routers.logs import router as logs_router


log = logging.getLogger("multicam")

# ─────────────────────────── FastAPI ──────────────────────────────────────────
app = FastAPI(title="Multi-Cam YOLO DeepStream", version="3.0")

app.include_router(cameras_router)
app.include_router(logs_router)
app.include_router(detections_router)


@app.on_event("startup")
def startup_event():
    log.info("Starting DeepStream camera pipeline...")
    start_pipelines()


@app.on_event("shutdown")
def shutdown_event():
    log.info("Stopping DeepStream camera pipeline...")
    stop_pipelines()


@app.get("/", response_class=HTMLResponse)
def root():
    links = "".join(
        f'<li><a href="/{cid}/live">/{cid}/live</a> - {cid} dashboard</li>'
        for cid in cam_ids
    )
    return HTMLResponse(f"""<html><body style="background:#111;color:#eee;font-family:monospace;padding:24px;">
<h2 style="color:#0f0">Multi-Cam YOLO DeepStream Server</h2>
<p>Backend: {CONFIG.get("pipeline_backend", "deepstream")}</p>
<ul>{links}
<li><a href="/log/live" style="color:#f80">/log/live</a> - Central log</li>
<li><a href="/detections">/detections</a> - JSON API</li>
<li><a href="/docs">/docs</a> - Swagger UI</li>
</ul></body></html>""")


# ─────────────────────────── Startup ──────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting FastAPI server on {CONFIG['host']}:{CONFIG['port']}")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="warning")
