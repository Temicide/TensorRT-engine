# TensorRT-engine Jetson Nano Setup

This repo runs a multi-camera YOLO server on Jetson Nano:

```text
RTSP cameras -> detector backend -> MJPEG dashboards + SSE + detection API
```

The current default backend in `config.py` is:

```python
"inference_backend": "tkdnn_darknet"
```

That means the server expects YOLOv4-tiny Darknet assets in:

```text
models/pipeline_2/darknet/
```

Expected files:

```text
yolov4-tiny.cfg
yolov4-tiny.weights
coco.names
```

If those files are already present after cloning, the next step is setting up
tkDNN and a small tkDNN JSON inference bridge on the Jetson Nano.

## 1. Jetson Prerequisites

Use a Jetson Nano image with JetPack installed. Confirm CUDA, TensorRT, and
OpenCV are available:

```bash
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import tensorrt as trt; print(trt.__version__)"
which nvcc || true
```

Install common build/runtime packages:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  curl \
  libopencv-dev \
  libyaml-cpp-dev \
  libeigen3-dev \
  python3-pip \
  python3-venv
```

## 2. Clone This Repo

```bash
git clone <this-repo-url>
cd TensorRT-engine
```

Check that the YOLOv4-tiny Darknet files exist:

```bash
ls -lh models/pipeline_2/darknet/
```

If they are missing:

```bash
cd models/pipeline_2
./download_yolov4_tiny_darknet.sh
cd ../..
```

## 3. Python Environment

Create a venv that can still see NVIDIA system packages such as TensorRT:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install fastapi uvicorn numpy opencv-python requests
```

If `opencv-python` causes issues on Jetson, remove it and use the JetPack
system OpenCV package instead:

```bash
python3 -m pip uninstall -y opencv-python
python3 -c "import cv2; print(cv2.__version__)"
```

## 4. Build tkDNN

Build tkDNN on the Jetson Nano:

```bash
cd ~
git clone https://github.com/ceccocats/tkDNN.git
cd tkDNN
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
```

Use `-j2` on Nano to reduce memory pressure. If the build runs out of memory,
enable swap or build with `make -j1`.

## 5. Export/Prepare YOLOv4-tiny for tkDNN

tkDNN does not use ONNX directly. Use the Darknet files:

```text
models/pipeline_2/darknet/yolov4-tiny.cfg
models/pipeline_2/darknet/yolov4-tiny.weights
models/pipeline_2/darknet/coco.names
```

Follow tkDNN's Darknet export workflow:

```text
https://github.com/ceccocats/tkDNN/blob/master/docs/exporting_weights.md
https://github.com/ceccocats/tkDNN/blob/master/docs/demo.md
```

For Jetson Nano, use YOLOv4-tiny rather than full YOLOv4.

## 6. Build or Install the tkDNN JSON Bridge

The Python server cannot call tkDNN C++ directly with only `.cfg` and
`.weights`. It expects a small executable bridge configured at:

```python
CONFIG["tkdnn"]["command"]
```

Expected command contract:

```bash
tkdnn_json_infer \
  --cfg models/pipeline_2/darknet/yolov4-tiny.cfg \
  --weights models/pipeline_2/darknet/yolov4-tiny.weights \
  --names models/pipeline_2/darknet/coco.names \
  --image /tmp/frame.jpg \
  --conf 0.25 \
  --iou 0.45
```

Expected stdout:

```json
[{"class_id":2,"class_name":"car","confidence":0.91,"bbox_xyxy":[10,20,120,220]}]
```

After the bridge is built, make it executable and test it manually:

```bash
chmod +x /path/to/tkdnn_json_infer
/path/to/tkdnn_json_infer \
  --cfg models/pipeline_2/darknet/yolov4-tiny.cfg \
  --weights models/pipeline_2/darknet/yolov4-tiny.weights \
  --names models/pipeline_2/darknet/coco.names \
  --image /path/to/test.jpg \
  --conf 0.25 \
  --iou 0.45
```

Then edit `config.py`:

```python
"tkdnn": {
    "bridge_mode": "persistent_command",
    "command": "/path/to/tkdnn_json_infer",
    "timeout_sec": 60.0,
}
```

Keep the existing `cfg`, `weights`, and `names` paths unless you moved the
model files. Use the top-level `max_detections` setting to cap drawn/logged
detections per frame.

On Jetson Nano, test one manual single-shot bridge run before starting all
cameras:

```bash
time /path/to/tkdnn_json_infer \
  --rt /home/ta/tkDNN/build/yolo4tiny_fp16.rt \
  --cfg models/pipeline_2/darknet/yolov4-tiny.cfg \
  --weights models/pipeline_2/darknet/yolov4-tiny.weights \
  --names models/pipeline_2/darknet/coco.names \
  --image /path/to/test.jpg \
  --conf 0.25 \
  --iou 0.45
```

The server uses `bridge_mode="persistent_command"` by default. That starts
`tkdnn_json_infer --server` once, keeps PyCUDA/TensorRT and the `.rt` file
loaded, then sends frame requests over stdin/stdout. If the first request times
out, raise `CONFIG["tkdnn"]["timeout_sec"]`; after startup, steady-state frames
should be much faster than the manual single-shot runtime.

## 7. Configure Cameras

Edit `config.py`:

```python
"active_cameras": ["cam1"],
"cameras": {
    "cam1": "rtsp://...",
}
```

Start with one camera first. After it works, set:

```python
"active_cameras": None
```

to enable all configured cameras.

## 8. Run the Server

From the repo root:

```bash
source venv/bin/activate
python3 server.py
```

Open from another machine on the same network:

```text
http://<jetson-ip>:8000/
http://<jetson-ip>:8000/cam1/live
http://<jetson-ip>:8000/log/live
http://<jetson-ip>:8000/detections
```

## 9. Fallback: Existing TensorRT Engine Backend

If you want to bypass tkDNN and use the existing Python TensorRT `.engine`
runtime, edit `config.py`:

```python
"inference_backend": "tensorrt_engine",
"model_path": "models/pipeline_1/exports/yolo26n_fp16.engine",
"imgsz": 640,
```

Build the engine on the Jetson:

```bash
cd models/pipeline_1
./jetson_build_engines_from_onnx.sh
cd ../..
```

Then run:

```bash
python3 server.py
```

## Troubleshooting

If startup says Darknet files are missing, run:

```bash
cd models/pipeline_2
./download_yolov4_tiny_darknet.sh
```

If startup says `CONFIG["tkdnn"]["command"]` is empty, tkDNN is not connected to
the Python server yet. Build/provide the `tkdnn_json_infer` executable and set
its path in `config.py`.

If logs say `tkDNN persistent bridge timed out after ...`, keep
`"active_cameras": ["cam1"]` and raise `CONFIG["tkdnn"]["timeout_sec"]` for the
first request. If logs say the bridge exited, read the captured stderr in the
same log line; it usually identifies a TensorRT, PyCUDA, `.rt`, or plugin-load
problem.

If RTSP capture is unstable, keep this in `config.py`:

```python
"enable_gstreamer": False
```

and start with one camera:

```python
"active_cameras": ["cam1"]
```
