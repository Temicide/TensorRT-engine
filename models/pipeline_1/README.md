# Vehicle Metadata Pipeline for Jetson Nano

Note: the production server path has moved to DeepStream. See
`../../README_DEEPSTREAM_NANO.md` for the RTSP -> nvv4l2decoder -> nvstreammux
-> nvinfer runtime. The scripts in this folder are kept for model export,
legacy demos, and vehicle metadata experiments.

This folder contains the files needed to run the vehicle metadata pipeline on Jetson Nano.

Pipeline:

```text
Camera stream / video
-> YOLO26n detection + tracking
-> crop vehicle
-> EfficientNet-B0 brand classifier
-> OpenCV color estimation
-> color majority vote per vehicle_id
-> JSONL metadata output
```

Output example:

```json
{"timestamp": "2026-05-19 10:30:00", "vehicle_id": 101, "brand": "Toyota", "color": "White"}
```

## Files

Model/export files:

```text
exports/yolo26n_opset12.onnx
exports/efficientnetb0_brand_opset12.onnx
exports/efficientnetb0_brand_opset12.labels.json
yolo26n.pt
```

Runtime code:

```text
vehicle_metadata_pipeline.py
vehicle_color_extractor.py
run_demo_video_realtime.py
train_efficientnetb0_brand.py
train_efficientnetb0_mtl.py
render_tracking_log_to_video.py
```

Build/export helper scripts:

```text
jetson_build_engines_from_onnx.sh
export_brand_onnx.py
export_yolo_onnx.py
```

For the DeepStream server, export the detector ONNX without NMS/post-processing:

```bash
python3 export_yolo_onnx.py --weights yolo26n.pt --imgsz 640 --opset 12
```

The exporter saves `exports/yolo26n_opset12.onnx` by default and rejects graphs
that still contain `TopK`, `Mod`, or `NonMaxSuppression`.

## Build TensorRT Engines On Jetson

Build `.engine` files on the Jetson Nano itself. Do not build `.engine` on Windows and copy it to Jetson.

```bash
chmod +x jetson_build_engines_from_onnx.sh
./jetson_build_engines_from_onnx.sh
```

Default FP32 outputs:

```text
.runtime/deepstream/yolov8n_b1_gpu0_fp32.engine
exports/efficientnetb0_brand_fp32.engine
```

After FP32 works, test FP16 explicitly:

```bash
USE_FP16=1 ./jetson_build_engines_from_onnx.sh
```

The runtime supports these brand classifier formats:

```text
.pt
.onnx
.engine
```

Default brand classifier:

```text
exports/efficientnetb0_brand_opset12.onnx
```

After building TensorRT, run with the brand engine:

```bash
--classifier exports/efficientnetb0_brand_fp32.engine
```

The detector can also use either PyTorch or TensorRT:

```bash
--detector yolo26n.pt
--detector exports/yolo26n_fp32.engine
```

## Install Runtime Dependencies

Use the environment already prepared for Jetson if available. Minimum Python packages:

```bash
pip3 install ultralytics opencv-python numpy pillow torch torchvision
```

On Jetson, prefer NVIDIA-provided PyTorch/torchvision wheels that match JetPack.

## Run Realtime

Set camera URL:

```bash
export CAMERA_URL='http://USER:PASSWORD@CAMERA_IP:PORT/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1'
```

Run headless with browser video link:

```bash
python3 vehicle_metadata_pipeline.py \
  --source "$CAMERA_URL" \
  --detector exports/yolo26n_fp32.engine \
  --classifier exports/efficientnetb0_brand_fp32.engine \
  --view \
  --print-json \
  --output live_vehicle_metadata.jsonl
```

The command prints a link like this:

```text
annotated video link: http://<jetson-ip>:8080/
```

Open that link from another computer on the same network. The Jetson does not open a local UI window.

Run with browser link and MP4 recording:

```bash
python3 vehicle_metadata_pipeline.py \
  --source "$CAMERA_URL" \
  --detector exports/yolo26n_fp32.engine \
  --classifier exports/efficientnetb0_brand_fp32.engine \
  --view \
  --save-video \
  --video-output live_annotated.mp4 \
  --print-json \
  --output live_vehicle_metadata.jsonl
```

Stop with `Ctrl+C` in the terminal.

To change the browser link port:

```bash
--web-port 8090
```

## Run Demo Video Realtime

Put a demo video file next to this README, for example:

```text
Test Video.mp4
```

Run the demo as browser-link video:

```bash
python3 run_demo_video_realtime.py \
  --source "Test Video.mp4" \
  --web-port 8080 \
  --output demo_vehicle_metadata.jsonl
```

Open:

```text
http://<jetson-ip>:8080/
```

The demo uses `--realtime-playback`, so a video file plays at its source FPS instead of running as fast as possible.

Save demo MP4 too:

```bash
python3 run_demo_video_realtime.py \
  --source "Test Video.mp4" \
  --save-video \
  --video-output demo_vehicle_metadata_annotated.mp4 \
  --output demo_vehicle_metadata.jsonl
```

## Performance Settings

Default detection rate is 30 FPS:

```bash
--detect-fps 30
```

If Jetson Nano lags, use 5 FPS or 3 FPS and the faster HSV color method:

```bash
python3 vehicle_metadata_pipeline.py \
  --source "$CAMERA_URL" \
  --detect-fps 5 \
  --color-method hsv \
  --view \
  --print-json \
  --output live_vehicle_metadata.jsonl
```

You can also manually skip frames:

```bash
--vid-stride 10
```

## Color Logic

Vehicle color is not a neural-network model. It uses OpenCV image processing:

- center body crop
- HSV/brightness rules for white/black/silver/gray
- K-Means or HSV rule-based color estimation
- rolling majority vote per `vehicle_id`

Default:

```bash
--color-method kmeans
--color-vote-window 15
```

For daytime-only operation, current thresholds are tuned to reduce:

- white detected as gray
- black detected as gray

Debug color crops:

```bash
python3 vehicle_metadata_pipeline.py \
  --source "$CAMERA_URL" \
  --save-color-crops color_debug \
  --include-confidence \
  --output color_debug.jsonl
```

## Metadata Behavior

Default behavior sends/logs metadata every processed detection:

```bash
--emit-mode every_detection
```

This means the same `vehicle_id` can appear multiple times. This is intentional so later frames can correct earlier brand/color mistakes.

To log only once per tracked vehicle:

```bash
--emit-mode first_seen
```

## Notes

- `.engine` files are hardware/version specific.
- `yolo26n.pt` is included as fallback if the Jetson team wants to export directly with Ultralytics on-device.
- `exports/efficientnetb0_brand_opset12.labels.json` maps brand output indices to class names.
- Color does not require export or quantization because it is OpenCV-only.
