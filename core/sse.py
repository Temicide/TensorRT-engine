import asyncio
import json

from core.state import global_sse_subscribers, per_cam_sse_subscribers, sse_sub_lock


def _safe_put(queue: asyncio.Queue, data: str) -> None:
    try:
        queue.put_nowait(data)
    except asyncio.QueueFull:
        pass


def _current_loop():
    # Jetson Nano / JetPack 4 commonly runs Python 3.6, which does not have
    # asyncio.get_running_loop(). Inside these async generators, get_event_loop()
    # returns the active uvicorn loop.
    return asyncio.get_event_loop()


# SSE is the live event broadcasting to the clients
# ─────────────────────────── SSE broadcast ────────────────────────────────────
def broadcast_sse(cam_id, record):
    data = json.dumps(record)
    with sse_sub_lock:
        subscribers = list(per_cam_sse_subscribers.get(cam_id, ()))
        subscribers.extend(global_sse_subscribers)

    for loop, q in subscribers:
        try:
            loop.call_soon_threadsafe(_safe_put, q, data)
        except RuntimeError:
            pass

# ── SSE per camera ─────────────────────────────────────────────────────────────
async def cam_sse_gen(cam_id):
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscriber = (_current_loop(), q)
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
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    subscriber = (_current_loop(), q)
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
