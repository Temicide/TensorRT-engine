from typing import Optional
from fastapi import APIRouter, Query
from core.sse import broadcast_sse
from core.state import (
    MAX_LOG,
    detection_log,
    detection_log_lock,
    per_cam_sse_subscribers,
)

router = APIRouter()

# ── Detections query API ───────────────────────────────────────────────────────
@router.get("/detections")
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


@router.post("/detections")
async def post_detection(record: dict):
    with detection_log_lock:
        detection_log.append(record)
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)
    cam_id = record.get("camera_id", "external")
    if cam_id in per_cam_sse_subscribers:
        broadcast_sse(cam_id, record)
    return {"ok": True}