import asyncio
import json

from core.state import global_sse_subscribers, per_cam_sse_subscribers, sse_sub_lock

# SSE is the live event broadcasting to the clients
# ─────────────────────────── SSE broadcast ────────────────────────────────────
def broadcast_sse(cam_id, record):
    data = json.dumps(record)
    with sse_sub_lock:
        # per-camera subscribers
        for q in per_cam_sse_subscribers[cam_id]:
            try: q.put_nowait(data)
            except asyncio.QueueFull: pass
        # global subscribers
        for q in global_sse_subscribers:
            try: q.put_nowait(data)
            except asyncio.QueueFull: pass

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