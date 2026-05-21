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
  -> sink or nvstreamdemux -> per-camera appsink
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
- Batch size must match the active camera count. The checked-in config starts
  one active RTSP stream for debugging.
- The checked-in runtime config uses FP16 (`network-mode=2`) with the engine
  filename produced by DeepStream-Yolo on DS6. If FP16 fails on your Nano stack, switch to FP32
  (`network-mode=0`) and remove the stale runtime engine.

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
ONNX model:        models/pipeline_beta/yolov8n(1).onnx
Engine output:     model_b1_gpu0_fp16.engine
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

## Install DeepStream Python Bindings (`pyds`)

This server imports `pyds` to read DeepStream metadata from Python. `pyds` is
not provided by this repository; install the wheel that matches the DeepStream
SDK on the Nano.

Check the installed DeepStream release first:

```bash
deepstream-app --version-all
python3 --version
uname -m
```

For DeepStream 6.0 on Jetson Nano / aarch64:

```bash
python3 -m pip install --user \
  https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.0/pyds-1.1.0-py3-none-linux_aarch64.whl
```

For DeepStream 6.0.1 on Jetson Nano / aarch64:

```bash
python3 -m pip install --user \
  https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.1/pyds-1.1.1-py3-none-linux_aarch64.whl
```

Verify the import with the same Python environment used for `server.py`:

```bash
python3 -c "import pyds; print('pyds OK')"
```

If you run the server from a virtualenv, create it with system site packages so
Jetson/DeepStream Python modules remain visible:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python3 -c "import gi, pyds; print('DeepStream Python imports OK')"
```

## Export YOLO ONNX

Preferred portable flow:

```text
YOLO .pt -> ONNX -> copy ONNX to Jetson Nano -> nvinfer builds .engine
```

The configured DeepStream detector path is:

```text
models/pipeline_beta/yolov8n(1).onnx
```

To re-export from a YOLOv8n `.pt` file, write to a stable path and update
`CONFIG["deepstream"]["onnx_model_path"]` if you change the filename:

```bash
cd models/pipeline_1
python3 export_yolo_onnx.py \
  --weights yolov8n.pt \
  --imgsz 640 \
  --opset 12 \
  --output ../../models/pipeline_beta/yolov8n_deepstream_opset12.onnx
```

Do not pass `--nms` for the DeepStream server. The DeepStream-YOLO parser
expects raw detector output and performs bbox decode/NMS itself. If TensorRT
logs an error around `TopK`, `Mod`, `NonMaxSuppression`, or "Plugin not found",
the ONNX was exported with post-processing included; regenerate it with the
command above and delete the stale engine before retrying.

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
rm -f model_b1_gpu0_fp16.engine
python3 server.py
```

Alternative: build with `trtexec` on the same Nano:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx='models/pipeline_beta/yolov8n(1).onnx' \
  --saveEngine=model_b1_gpu0_fp32.engine \
  --workspace=1024 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

Only after FP32 works, test FP16:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx='models/pipeline_beta/yolov8n(1).onnx' \
  --saveEngine=model_b1_gpu0_fp16.engine \
  --fp16 \
  --workspace=1024 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640
```

If you switch precision, update `engine_path` and `network_mode` together in `config.py`.
Jetson Nano does not have Orin-style Tensor Cores, so measure rather than
assuming a large speedup.

## Run Server

Edit `config.py`:

- `active_cameras`: set the cameras to start, for example all configured
  cameras or only `["cam1"]` while debugging.
- `cameras`: set each RTSP URI.
- `deepstream.onnx_model_path`
- `deepstream.engine_path`
- `deepstream.labels_path`
- `deepstream.custom_parser_path`
- `deepstream.input_width` / `input_height`
- `deepstream.batch_size`: must match `len(active_cameras)`.
- `deepstream.workspace_size`
- `deepstream.sink_type`: `fakesink`, `egl`, or `appsink`
- `deepstream.enable_mjpeg_output`: only true with `sink_type="appsink"`
- `deepstream.rtsp_preflight`: keep true while debugging source URLs. It sends a
  lightweight RTSP `OPTIONS` request before DeepStream starts, so wrong
  HTTP/FastAPI ports fail fast instead of looping inside `rtspsrc`.

Run:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip3 install fastapi uvicorn requests numpy
# Optional, only for sink_type="appsink" / MJPEG output:
pip3 install opencv-python
python3 tools/validate-deepstream-env.py
python3 server.py
```

The DeepStream pipeline starts from FastAPI startup/shutdown hooks, so
`uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1` also works. Keep
workers at `1`; each worker process would otherwise create its own DeepStream
pipeline.

Open:

```text
http://<jetson-ip>:8000/
http://<jetson-ip>:8000/log/live
http://<jetson-ip>:8000/detections
```

For maximum Nano headroom, keep the defaults: `sink_type="fakesink"` and
`enable_mjpeg_output=False`. In this mode the pipeline probes metadata directly
after `nvinfer` or `nvtracker` and skips OSD/color conversion. Use `appsink`
only when the existing MJPEG endpoint is required; it copies annotated frames to
CPU and JPEG-encodes them.

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

If the log shows `Could not receive message. (Parse error)` from `gstrtspsrc`,
the endpoint did not return a valid RTSP response. Check that the URL is a real
RTSP service and not an HTTP endpoint on the same host. The first response line
from the source should start with `RTSP/1.0`, not `HTTP/1.1`.

## Notes

- `configs/deepstream/primary_gie_yolov8n_nano.txt` is a reference config.
  At runtime, `core.deepstream_config` writes a generated nvinfer config to
  `/tmp/tensorrt_engine_primary_gie_yolov8n_nano.txt` using values from
  `config.py`.
- Checked-in `.engine` files should not be assumed valid for Nano unless they
  were built on this exact Nano environment. Runtime engines match
  `model_b*_gpu*_*.engine`, and `*.engine` is ignored by git.
- The separate `models/pipeline_1/vehicle_metadata_pipeline.py` script remains a
  legacy/demo workflow. The server path now consumes DeepStream metadata.
