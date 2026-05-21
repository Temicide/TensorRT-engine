# Pipeline 2: YOLOv4-tiny Darknet for Jetson Nano

This pipeline prepares YOLOv4-tiny assets in two forms:

```text
Darknet cfg + weights -> tkDNN Darknet exporter path
Darknet cfg + weights -> ONNX -> TensorRT engine path
```

Important: tkDNN does not consume ONNX directly. For tkDNN, keep the
`yolov4-tiny.cfg`, `yolov4-tiny.weights`, and `coco.names` files and export the
weights with the tkDNN Darknet workflow. The ONNX file is useful for your current
Python TensorRT runtime and for `trtexec` benchmarking.

## 1. Download Darknet YOLOv4-tiny assets

Run on the Jetson or on a machine with internet access:

```bash
cd models/pipeline_2
chmod +x download_yolov4_tiny_darknet.sh
./download_yolov4_tiny_darknet.sh
```

Expected files:

```text
darknet/yolov4-tiny.cfg
darknet/yolov4-tiny.weights
darknet/coco.names
```

## 2. Export ONNX

Install the export-only Python dependencies:

```bash
python3 -m pip install torch onnx
```

Export a static 416x416 ONNX model:

```bash
python3 export_yolov4_tiny_onnx.py \
  --cfg darknet/yolov4-tiny.cfg \
  --weights darknet/yolov4-tiny.weights \
  --output exports/yolov4_tiny_416_raw.onnx \
  --imgsz 416 \
  --opset 12
```

The ONNX outputs are raw YOLO feature maps, not post-NMS boxes:

```text
detect_0
detect_1
```

Your current `core/tensorrt_engine.py` decoder is built for YOLOv5/YOLOv8 style
outputs, so it will need a YOLOv4 decoder before this ONNX engine can replace
`yolov8n.engine` in the FastAPI app.

## 3. Build a TensorRT engine from ONNX

Run this on the Jetson Nano that will run inference:

```bash
chmod +x build_yolov4_tiny_onnx_engine_on_jetson.sh
./build_yolov4_tiny_onnx_engine_on_jetson.sh
```

Expected output:

```text
exports/yolov4_tiny_416_raw_fp16.engine
```

If FP16 fails on your JetPack/TensorRT version, edit the script and remove
`--fp16`.

## 4. tkDNN path

For tkDNN, do not use the ONNX file. Use tkDNN's Darknet exporter flow:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libopencv-dev libyaml-cpp-dev libeigen3-dev

git clone https://github.com/ceccocats/tkDNN.git
cd tkDNN
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
```

If CMake previously failed with `Could not find a package configuration file
provided by "yaml-cpp"`, install `libyaml-cpp-dev`, remove the stale CMake
cache with `rm -rf CMakeCache.txt CMakeFiles`, then run the `cmake` command
again.

Then export YOLOv4-tiny Darknet weights using the tkDNN documentation:

```text
https://github.com/ceccocats/tkDNN/blob/master/docs/exporting_weights.md
https://github.com/ceccocats/tkDNN/blob/master/docs/demo.md
```

tkDNN has a supported `Yolo4tiny` detector, which is the Jetson Nano-friendly
choice. Full YOLOv4 is much heavier on Nano.

## 5. Use from this FastAPI server

The server is configured through `config.py`:

```python
"inference_backend": "tkdnn_darknet",
"tkdnn": {
    "cfg": "models/pipeline_2/darknet/yolov4-tiny.cfg",
    "weights": "models/pipeline_2/darknet/yolov4-tiny.weights",
    "names": "models/pipeline_2/darknet/coco.names",
    "bridge_mode": "persistent_command",
    "command": "/path/to/tkdnn_json_infer",
    "timeout_sec": 60.0,
}
```

Use the top-level `max_detections` setting in `config.py` to cap drawn/logged
detections per frame.

The Python server cannot call the tkDNN C++ library directly with only
`yolov4-tiny.cfg` and `yolov4-tiny.weights`. It expects a small executable
bridge at `CONFIG["tkdnn"]["command"]`.

Expected bridge command:

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

The FastAPI server uses `bridge_mode="persistent_command"` by default. It
starts `tkdnn_json_infer --server` once, keeps PyCUDA/TensorRT and the `.rt`
engine loaded, then sends frame requests over stdin/stdout. If the first
request times out, raise `timeout_sec`; after startup, steady-state frames
should be much faster than a manual single-shot run.

Until that bridge exists, startup will fail with a clear configuration error.
To return to the previous Python TensorRT engine workflow, set:

```python
"inference_backend": "tensorrt_engine"
```
