# DeepStream Jetson Nano Runtime

This repository now uses a DeepStream/GStreamer runtime for the server path:

```text
RTSP source
  -> rtph264depay / h264parse
  -> nvv4l2decoder
  -> nvstreammux
  -> nvinfer
  -> optional nvtracker
  -> nvvideoconvert
  -> nvdsosd
  -> sink
```

The old OpenCV capture and manual PyCUDA/TensorRT inference modules remain only
as legacy reference code. `server.py` starts `core.deepstream_pipeline` and reads
detections from DeepStream metadata.

## Target

Use conservative Jetson Nano assumptions:

- Jetson Nano, not Orin or Xavier.
- JetPack 4.6.x.
- DeepStream 6.0 or 6.0.1.
- CUDA 10.2 on standard JetPack 4.6.x.
- Batch size 1 and one RTSP stream until the pipeline is validated.
- Start with FP32 (`network-mode=0`). Try FP16 (`network-mode=2`) only after FP32 works.

## Current Vanilla Pipeline That Was Replaced

- `core/camera_capture.py` used `cv2.VideoCapture` for RTSP frames.
- `core/pipeline.py` resized and normalized CPU frames with OpenCV/Numpy.
- `core/tensorrt_engine.py` deserialized a prebuilt `.engine`, managed PyCUDA buffers, decoded YOLO output, and ran CPU NMS.
- `core/pipeline.py` then updated `cam_state`, appended detection logs, broadcast SSE, and pushed detections to the external API.

The DeepStream path keeps the same business output flow but feeds it from a pad
probe over DeepStream metadata instead of raw Python inference results.

## Required Files

Configured in `config.py` under `CONFIG["deepstream"]`:

```text
ONNX model:        models/pipeline_1/exports/yolo26n_opset12.onnx
Engine output:     models/pipeline_1/exports/yolo26n_deepstream_nano_b1_fp32.engine
Labels:            configs/deepstream/labels_coco.txt
YOLO parser .so:   /opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
Primary GIE ref:   configs/deepstream/primary_gie_yolov8n_nano.txt
```

Missing pieces that must be built or copied on the Nano:

- A YOLOv8-compatible DeepStream custom parser source tree.
- The compiled `libnvdsinfer_custom_impl_Yolo.so`.
- A TensorRT engine generated on the same Jetson Nano, or no engine file so
  `nvinfer` can build it on first run.

Do not copy a `.engine` from Orin, desktop GPU, Colab, or another JetPack /
TensorRT / CUDA / DeepStream version.

## Check Jetson Versions

Run these on the Jetson Nano:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
deepstream-app --version-all
python3 -c "import tensorrt as trt; print(trt.__version__)"
/usr/src/tensorrt/bin/trtexec --version
```

For a standard Nano setup, expect JetPack 4.6.x and DeepStream 6.0/6.0.1.

## Export YOLO ONNX

Preferred portable flow:

```text
YOLO .pt -> ONNX -> copy ONNX to Jetson Nano -> nvinfer builds .engine
```

The repository already contains:

```text
models/pipeline_1/exports/yolo26n_opset12.onnx
```

To re-export:

```bash
cd models/pipeline_1
python3 export_yolo_onnx.py --weights yolo26n.pt --imgsz 640 --opset 12
```

If 640 is too slow on Nano, export 416 or 320 and update
`CONFIG["deepstream"]["input_width"]` / `input_height` plus the nvinfer config.

## Build YOLO Parser On Nano

Use a DeepStream-YOLO parser implementation that explicitly supports YOLOv8 and
DeepStream 6.0/6.0.1. Do not blindly use a latest branch intended for newer
DeepStream releases.

Typical build shape on JetPack 4.6.x:

```bash
cd /opt/nvidia/deepstream/deepstream-6.0/sources
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
# Pin a tag/commit compatible with DeepStream 6.0/6.0.1 before building.
cd nvdsinfer_custom_impl_Yolo
CUDA_VER=10.2 make -j"$(nproc)"
```

Then confirm the configured `.so` exists:

```bash
ls -l /opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

If your parser exposes different symbol names, update these in `config.py` or
`configs/deepstream/primary_gie_yolov8n_nano.txt`:

```text
parse-bbox-func-name=NvDsInferParseYolo
engine-create-func-name=NvDsInferYoloCudaEngineGet
```

## Build Or Regenerate Engine On Nano

Preferred: let `nvinfer` build the engine on first run.

```bash
rm -f models/pipeline_1/exports/yolo26n_deepstream_nano_b1_fp32.engine
python3 server.py
```

Alternative: build with `trtexec` on the same Nano:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/pipeline_1/exports/yolo26n_opset12.onnx \
  --saveEngine=models/pipeline_1/exports/yolo26n_deepstream_nano_b1_fp32.engine \
  --workspace=1024 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

Only after FP32 works, test FP16:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/pipeline_1/exports/yolo26n_opset12.onnx \
  --saveEngine=models/pipeline_1/exports/yolo26n_deepstream_nano_b1_fp16.engine \
  --fp16 \
  --workspace=1024 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

If you switch to FP16, update `engine_path` and `network_mode=2` in `config.py`.
Jetson Nano does not have Orin-style Tensor Cores, so measure rather than
assuming a large speedup.

## Run Server

Edit `config.py`:

- `active_cameras`: keep `["cam1"]` for first validation.
- `cameras["cam1"]`: set your RTSP URI.
- `deepstream.onnx_model_path`
- `deepstream.engine_path`
- `deepstream.labels_path`
- `deepstream.custom_parser_path`
- `deepstream.input_width` / `input_height`
- `deepstream.batch_size`
- `deepstream.workspace_size`
- `deepstream.sink_type`: `fakesink`, `egl`, or `appsink`
- `deepstream.enable_mjpeg_output`: only true with `sink_type="appsink"`

Run:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip3 install fastapi uvicorn requests numpy
python3 server.py
```

Open:

```text
http://<jetson-ip>:8000/
http://<jetson-ip>:8000/log/live
http://<jetson-ip>:8000/detections
```

For maximum Nano headroom, use `sink_type="fakesink"` and keep
`enable_mjpeg_output=False`. Use `appsink` only when the existing MJPEG endpoint
is required; it copies annotated frames to CPU and JPEG-encodes them.

## Validation Checklist

- RTSP connects with `rtspsrc`.
- Hardware decode uses `nvv4l2decoder`.
- `nvstreammux` receives the source and batch size is 1.
- `nvinfer` loads the ONNX or generated engine.
- The engine is created on this Jetson Nano.
- The YOLO parser `.so` loads successfully.
- Detections appear in DeepStream metadata.
- Bounding boxes are aligned with objects.
- Labels match `configs/deepstream/labels_coco.txt`.
- FPS is visible in `/log/live` or `/detections`.
- CPU usage is lower than the old OpenCV/Python inference loop.
- Memory usage is acceptable on Nano.
- Pipeline restarts after RTSP errors or EOS.
- If tracker is enabled, `track_id` appears and remains stable enough for the use case.

## Notes

- `configs/deepstream/primary_gie_yolov8n_nano.txt` is a reference config.
  At runtime, `core.deepstream_config` writes a generated nvinfer config to
  `/tmp/tensorrt_engine_primary_gie_yolov8n_nano.txt` using values from
  `config.py`.
- `models/pipeline_beta/yolov8n_fp16.engine` should not be assumed valid for
  Nano unless it was built on this exact Nano environment.
- The separate `models/pipeline_1/vehicle_metadata_pipeline.py` script remains a
  legacy/demo workflow. The server path now consumes DeepStream metadata.
