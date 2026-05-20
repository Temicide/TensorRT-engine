from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse

from core.sse import cam_sse_gen
from core.state import cam_state
from core.mjpeg import mjpeg_gen
from static.templates import cam_live_html

router = APIRouter()

@router.get("/cam{cam_num}/video")
def cam_video(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return Response("Camera not found", 404)
    return StreamingResponse(mjpeg_gen(cid),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/cam{cam_num}/stream")
async def cam_stream(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return Response("Camera not found", 404)
    return StreamingResponse(cam_sse_gen(cid), media_type="text/event-stream")

# ── loglive redirect ──────────────────────────────────────────────────────────
@router.get("/cam{cam_num}/loglive")
def cam_loglive_redirect(cam_num: int):
    """All /camN/loglive redirect to central /log/live"""
    return RedirectResponse(url="/log/live")

@router.get("/cam{cam_num}/live", response_class=HTMLResponse)
def cam_live(cam_num: int):
    cid = f"cam{cam_num}"
    if cid not in cam_state:
        return HTMLResponse("Camera not found", 404)
    return HTMLResponse(cam_live_html(cid))


