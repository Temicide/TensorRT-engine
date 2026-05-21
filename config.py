# ─────────────────────────── CONFIG (edit only here) ──────────────────────────
CONFIG = {
    # Runtime backend. The production server now uses DeepStream/nvinfer.
    # The legacy Python TensorRT wrapper is kept in the repository for reference,
    # but server.py no longer loads or executes TensorRT engines directly.
    "pipeline_backend": "deepstream",

    # Portable model artifact. Let DeepStream/TensorRT build the engine on the
    # Jetson Nano unless the .engine was generated on this exact Nano stack.
    "model_path":     "models/pipeline_beta/yolov8n(1).onnx",
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,
    "imgsz":          640,
    "mjpeg_fps":      8,
    "host":           "0.0.0.0",
    "port":           8000,
    "jetson_id":      "jetson-nano-01",
    "device_name":    "jetson-nano-01",
    "external_api":   "http://10.0.11.153:8080/api/v1/raw_data",

    # DeepStream is GPU-only on Jetson. If DeepStream/PyGObject/pyds cannot load,
    # startup stops instead of silently falling back to CPU inference.
    "gpu_required":   True,

    # Cameras to start. Keep deepstream.batch_size equal to this count.
    "active_cameras": ["cam1","cam2","cam3","cam4","cam5"],

    "cameras": {
        "cam1": "rtsp://10.0.11.153:8554/cctv06",
        "cam2": "rtsp://10.0.11.153:8554/cctv07",
        "cam3": "rtsp://10.0.11.153:8554/cctv10",
        "cam4": "rtsp://10.0.11.153:8554/cctv08",
        "cam5": "rtsp://10.0.11.153:8554/cctv09",
    },

    "deepstream": {
        # Target Jetson Nano baseline: JetPack 4.6.x + DeepStream 6.0/6.0.1.
        # Do not assume Orin/Xavier/Ampere behavior or DeepStream 7+ plugins.
        "rtsp_uri": None,  # None means use CONFIG["cameras"][active camera].
        "codec": "h264",  # h264 or h265.
        "rtsp_latency_ms": 200,
        "rtsp_tcp": True,
        "drop_on_latency": True,
        "reconnect_sec": 5,

        # Must be a raw YOLO detector export. Do not use an ONNX with NMS,
        # TopK, or Mod post-processing baked into the graph.
        # DeepStream-Yolo on DS6 serializes engines as model_b<batch>_gpu<id>_<precision>.engine
        # in the working directory. Keep the nvinfer model-engine-file pointed at
        # that path so restarts reuse the file instead of rebuilding.
        "onnx_model_path": "models/pipeline_beta/yolov8n(1).onnx",
        "engine_path": "model_b1_gpu0_fp16.engine",
        "labels_path": "configs/deepstream/labels_coco.txt",
        "custom_parser_path": "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so",
        "primary_gie_config_path": "configs/deepstream/primary_gie_yolov8n_nano.txt",

        "network_mode": 2,  # 0=FP32, 1=INT8, 2=FP16
        "batch_size": 1,
        "gpu_id": 0,
        "num_detected_classes": 80,
        "input_width": 640,
        "input_height": 640,
        "output_blob_names": "output0",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "topk": 300,
        "interval": 0,
        "workspace_size": 1024,

        # Streammux output size. Keep this near the camera aspect ratio; nvinfer
        # handles model input resizing with aspect-ratio padding.
        "mux_width": 1280,
        "mux_height": 720,
        "batched_push_timeout_us": 40000,

        # Optional tracker. Keep disabled until detection metadata is validated.
        "enable_tracker": False,
        "tracker_config_path": "configs/deepstream/tracker_nano_config.txt",

        # Sink options: fakesink, egl, appsink. Use fakesink for production
        # metadata/API mode. appsink preserves the MJPEG endpoint but copies
        # frames to CPU and costs Nano CPU time.
        "display": False,
        "sink_type": "fakesink",
        "enable_mjpeg_output": False,
        "jpeg_quality": 70,
    },
}
