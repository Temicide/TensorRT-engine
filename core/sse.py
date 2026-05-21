import asyncio
import json

from core.state import global_sse_subscribers, per_cam_sse_subscribers, sse_sub_lock

# SSE is the live event broadcasting to the clients
# ─────────────────────────── SSE broadcast ────────────────────────────────────
def broadcast_sse(cam_id, record):
    data = json.dumps(record)
    with sse_sub_lock:
        per_cam_targets = list(per_cam_sse_subscribers[cam_id])
        global_targets = list(global_sse_subscribers)

    for loop, q in per_cam_targets + global_targets:
        try:
            loop.call_soon_threadsafe(_put_nowait, q, data)
        except RuntimeError:
            pass


def _put_nowait(q, data):
    try:
        q.put_nowait(data)
    except asyncio.QueueFull:
        pass

# ── SSE per camera ─────────────────────────────────────────────────────────────
async def cam_sse_gen(cam_id):
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscriber = (loop, q)
    with sse_sub_lock:
        per_cam_sse_subscribers[cam_id].append(subscriber)
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
            if subscriber in per_cam_sse_subscribers[cam_id]:
                per_cam_sse_subscribers[cam_id].remove(subscriber)

# ── SSE global (all cameras) ───────────────────────────────────────────────────
async def global_sse_gen():
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    subscriber = (loop, q)
    with sse_sub_lock:
        global_sse_subscribers.append(subscriber)
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
            if subscriber in global_sse_subscribers:
                global_sse_subscribers.remove(subscriber)
