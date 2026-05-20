# ─────────────────────────── CONFIG (edit only here) ──────────────────────────
CONFIG = {
    # TensorRT engine built for this exact Jetson / TensorRT / CUDA version.
    # Example: /usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
    "model_path":     "yolov8n.engine",
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "imgsz":          224,
    "mjpeg_fps":      20,
    "host":           "0.0.0.0",
    "port":           8000,
    "jetson_id":      "jetson-nano-01",
    "device_name":    "jetson-nano-01",
    "external_api":   "http://10.0.11.153:8080/api/v1/raw_data",

    # TensorRT engine inference is GPU-only. If TensorRT/PyCUDA cannot load, startup stops.
    "gpu_required":   True,

    # Capture stability controls for Jetson Nano.
    # If GStreamer probing crashes or fails, keep this False and use OpenCV/FFmpeg TCP.
    "enable_gstreamer": False,
    # Start capture threads one by one, not all at the same millisecond.
    "capture_startup_stagger_sec": 1.5,
    # For debugging, set to ["cam1"] first. Use None to enable all cameras.
    "active_cameras": None,

    "cameras": {
        "cam1": "rtsp://10.0.11.153:8554/mock1",
        "cam2": "rtsp://10.0.11.153:8554/mock5",
        "cam3": "rtsp://10.0.11.153:8554/mock9",
        "cam4": "rtsp://10.0.11.153:8554/cctv08",
        "cam5": "rtsp://10.0.11.153:8554/cctv09",
    },
}