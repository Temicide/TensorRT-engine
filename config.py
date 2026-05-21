# ─────────────────────────── CONFIG (edit only here) ──────────────────────────
CONFIG = {
    # Inference backend:
    #   "tensorrt_engine" -> existing Python TensorRT .engine runtime.
    #   "tkdnn_darknet"   -> tkDNN/Darknet bridge using models/pipeline_2/darknet.
    "inference_backend": "tkdnn_darknet",

    # TensorRT engine built for this exact Jetson / TensorRT / CUDA version.
    # Example: /usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
    "model_path":     "yolov8n.engine",
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "max_detections": 50,
    "imgsz":          416,
    "mjpeg_fps":      20,
    "host":           "0.0.0.0",
    "port":           8000,
    "jetson_id":      "jetson-nano-01",
    "device_name":    "jetson-nano-01",
    "external_api":   "http://10.0.11.153:8080/api/v1/raw_data",

    # TensorRT engine inference is GPU-only. If TensorRT/PyCUDA cannot load, startup stops.
    "gpu_required":   True,

    # tkDNN/Darknet detector configuration.
    #
    # tkDNN does not run ONNX directly. It uses Darknet cfg/weights during its
    # export/build workflow, then runs a tkDNN/TensorRT runtime. The Python
    # server needs a small executable bridge for per-frame inference.
    #
    # Expected bridge contract for "image_command" mode:
    #   <command> --cfg <cfg> --weights <weights> --names <names>
    #             --image <jpg> --conf <float> --iou <float>
    # It must print JSON to stdout:
    #   [{"class_id":2,"class_name":"car","confidence":0.91,
    #     "bbox_xyxy":[10,20,120,220]}]
    "tkdnn": {
        "darknet_dir": "models/pipeline_2/darknet",
        "cfg": "models/pipeline_2/darknet/yolov4-tiny.cfg",
        "weights": "models/pipeline_2/darknet/yolov4-tiny.weights",
        "names": "models/pipeline_2/darknet/coco.names",
        "rt": "/home/ta/tkDNN/build/yolo4tiny_fp16.rt",
        "bridge_mode": "persistent_command",
        "command": "/home/ta/hardteam_ws/tensorrt/chi_ws/tools/tkdnn_json_infer",
        # persistent_command keeps PyCUDA/TensorRT and the .rt file loaded.
        # The first request may still include one-time TensorRT startup cost.
        "timeout_sec": 60.0,
    },

    # Capture stability controls for Jetson Nano.
    # Use GStreamer for RTSP capture.
    "enable_gstreamer": True,
    # If True, do not fall back to OpenCV/FFmpeg when GStreamer fails.
    "require_gstreamer": True,
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
