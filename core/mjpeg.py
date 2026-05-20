import time
from config import CONFIG
from core.state import cam_state

MJPEG_DELAY = 1.0 / CONFIG["mjpeg_fps"]

def mjpeg_gen(cam_id):
    state = cam_state[cam_id]
    while True:
        with state["lock"]:
            jpg = state["frame_jpg"]
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(MJPEG_DELAY)