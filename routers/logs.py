from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from core.sse import global_sse_gen
from static.templates import log_live_html

router = APIRouter()

@router.get("/log/stream")
async def log_stream():
    return StreamingResponse(global_sse_gen(), media_type="text/event-stream")

@router.get("/log/live", response_class=HTMLResponse)
def log_live():
    return HTMLResponse(log_live_html())