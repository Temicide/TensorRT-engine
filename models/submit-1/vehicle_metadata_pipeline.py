from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))

from train_efficientnetb0_mtl import EfficientNetB0MultiTask  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from vehicle_color_extractor import estimate_vehicle_color  # noqa: E402


DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def normalize_weights_name(weights: str) -> str:
    return "yolo26n.pt" if weights == "yolov26n.pt" else weights


def color_for_track(track_id: int) -> tuple[int, int, int]:
    value = int(track_id) * 2654435761 % 255
    return (
        int((value * 3 + 80) % 255),
        int((value * 7 + 120) % 255),
        int((value * 11 + 160) % 255),
    )


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, text_h + 8)
    cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 8, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


class MjpegState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpeg: bytes | None = None

    def update(self, frame, quality: int) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        with self.lock:
            self.frame_jpeg = encoded.tobytes()

    def snapshot(self) -> bytes | None:
        with self.lock:
            return self.frame_jpeg


def start_web_view(args: argparse.Namespace):
    state = MjpegState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = (
                    "<!doctype html><html><head><title>Vehicle Metadata</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                    "img{width:100vw;height:100vh;object-fit:contain}</style></head>"
                    "<body><img src='/stream'></body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                frame = state.snapshot()
                if frame is None:
                    threading.Event().wait(0.05)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    threading.Event().wait(1.0 / max(args.web_fps, 1.0))
                except (BrokenPipeError, ConnectionResetError):
                    break

    server = ThreadingHTTPServer((args.web_host, args.web_port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host_for_print = args.web_host if args.web_host not in ("0.0.0.0", "") else "<jetson-ip>"
    print(f"annotated video link: http://{host_for_print}:{args.web_port}/", flush=True)
    return server, state


def video_fps(source: str) -> float:
    capture = cv2.VideoCapture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return fps if fps > 0 else 30.0


def resolve_vid_stride(args: argparse.Namespace, fps: float) -> int:
    if args.vid_stride is not None:
        return max(1, int(args.vid_stride))
    if args.detect_fps is None or args.detect_fps <= 0:
        return 1
    return max(1, int(round(fps / args.detect_fps)))


class EfficientNetB0Brand(nn.Module):
    def __init__(self, num_brands: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.brand_head = nn.Linear(in_features, num_brands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.brand_head(self.backbone(x))


def load_brand_labels(labels_path: Path) -> list[str]:
    candidates = [
        labels_path.with_suffix(".labels.json"),
        labels_path.parent / labels_path.name.replace("_fp16.engine", "_opset12.labels.json"),
        labels_path.parent / labels_path.name.replace(".engine", ".labels.json"),
        labels_path.parent / labels_path.name.replace(".onnx", ".labels.json"),
    ]
    candidates.extend(sorted(labels_path.parent.glob("*brand*.labels.json")))

    for candidate in candidates:
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "brand_classes" in data:
            return data["brand_classes"]
        if isinstance(data, list):
            return data

    raise FileNotFoundError(f"Could not find brand labels JSON next to {labels_path}")


def preprocess_brand_numpy(crop_bgr, imgsz: int = 224) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(crop_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits.reshape(-1).astype(np.float32)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def torch_load_checkpoint(path: Path, device: torch.device):
    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class TorchBrandClassifier:
    def __init__(self, checkpoint_path: Path, device: torch.device, imgsz: int):
        checkpoint = torch_load_checkpoint(checkpoint_path, device)
        self.brand_classes = checkpoint["brand_classes"]
        self.device = device
        self.transform = preprocess_crop(imgsz)

        if "color_classes" in checkpoint:
            self.model = EfficientNetB0MultiTask(
                num_brands=len(self.brand_classes),
                num_colors=len(checkpoint["color_classes"]),
                pretrained=False,
            )
        else:
            self.model = EfficientNetB0Brand(num_brands=len(self.brand_classes))

        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, crop_bgr) -> dict[str, object]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        brand_logits = output[0] if isinstance(output, tuple) else output
        brand_probs = torch.softmax(brand_logits, dim=1)[0]
        brand_idx = int(brand_probs.argmax().item())
        return {
            "brand": self.brand_classes[brand_idx],
            "brand_confidence": round(float(brand_probs[brand_idx].item()), 4),
        }


class OnnxBrandClassifier:
    def __init__(self, checkpoint_path: Path, imgsz: int):
        self.brand_classes = load_brand_labels(checkpoint_path)
        self.imgsz = imgsz
        self.net = cv2.dnn.readNetFromONNX(str(checkpoint_path))

    def predict(self, crop_bgr) -> dict[str, object]:
        tensor = preprocess_brand_numpy(crop_bgr, self.imgsz)
        self.net.setInput(tensor)
        logits = self.net.forward()
        probs = softmax_numpy(logits)
        brand_idx = int(np.argmax(probs))
        return {
            "brand": self.brand_classes[brand_idx],
            "brand_confidence": round(float(probs[brand_idx]), 4),
        }


class TensorRTBrandClassifier:
    def __init__(self, checkpoint_path: Path, imgsz: int):
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise ImportError("TensorRT .engine runtime requires `tensorrt` and `pycuda` on Jetson.") from exc

        self.trt = trt
        self.cuda = cuda
        self.brand_classes = load_brand_labels(checkpoint_path)
        self.imgsz = imgsz
        self.logger = trt.Logger(trt.Logger.WARNING)

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(checkpoint_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {checkpoint_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_idx = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
        self.output_idx = next(i for i in range(engine.num_bindings) if not engine.binding_is_input(i))

        if -1 in tuple(engine.get_binding_shape(self.input_idx)):
            self.context.set_binding_shape(self.input_idx, (1, 3, imgsz, imgsz))

        self.input_shape = tuple(self.context.get_binding_shape(self.input_idx))
        self.output_shape = tuple(self.context.get_binding_shape(self.output_idx))
        self.input_dtype = trt.nptype(engine.get_binding_dtype(self.input_idx))
        self.output_dtype = trt.nptype(engine.get_binding_dtype(self.output_idx))

        self.host_input = np.empty(self.input_shape, dtype=self.input_dtype)
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        self.bindings = [0] * engine.num_bindings
        self.bindings[self.input_idx] = int(self.device_input)
        self.bindings[self.output_idx] = int(self.device_output)

    def predict(self, crop_bgr) -> dict[str, object]:
        tensor = preprocess_brand_numpy(crop_bgr, self.imgsz).astype(self.input_dtype, copy=False)
        np.copyto(self.host_input, tensor)
        self.cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        probs = softmax_numpy(self.host_output)
        brand_idx = int(np.argmax(probs))
        return {
            "brand": self.brand_classes[brand_idx],
            "brand_confidence": round(float(probs[brand_idx]), 4),
        }


def load_brand_classifier(checkpoint_path: Path, device: torch.device, imgsz: int):
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".pt":
        return TorchBrandClassifier(checkpoint_path, device, imgsz=imgsz)
    if suffix == ".onnx":
        return OnnxBrandClassifier(checkpoint_path, imgsz=imgsz)
    if suffix == ".engine":
        return TensorRTBrandClassifier(checkpoint_path, imgsz=imgsz)
    raise ValueError(f"Unsupported classifier format: {checkpoint_path}")


def preprocess_crop(imgsz: int):
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def crop_box(frame, box, pad: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_color_debug_crop(args: argparse.Namespace, crop, frame_idx: int, track_id: int, color: str) -> None:
    if not args.save_color_crops:
        return
    output_dir = Path(args.save_color_crops)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_color = color.replace(" ", "_").replace("-", "_")
    cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}_id_{track_id}_{safe_color}.jpg"), crop)


def update_color_vote(
    color_history_by_id: dict[int, list[str]],
    track_id: int,
    raw_color: str,
    window: int,
) -> tuple[str, float, int]:
    history = color_history_by_id.setdefault(track_id, [])
    history.append(raw_color)
    if window > 0:
        del history[:-window]

    counts = Counter(history)
    max_votes = max(counts.values())
    tied = {color for color, votes in counts.items() if votes == max_votes}

    # If two colors tie, prefer the most recent tied observation. This keeps the
    # vote responsive when the first few crops were mostly windshield/shadow.
    for color in reversed(history):
        if color in tied:
            winner = color
            break

    return winner, round(max_votes / len(history), 4), len(history)


def classify_brand(crop_bgr, classifier):
    return classifier.predict(crop_bgr)


def parse_line(raw: str | None):
    if not raw:
        return None
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != 4:
        raise ValueError("--line must be x1,y1,x2,y2")
    return tuple(values)


def side_of_line(point, line) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossed_line(track_id: int, center, line, previous_centers: dict[int, tuple[float, float]]) -> bool:
    previous = previous_centers.get(track_id)
    previous_centers[track_id] = center
    if previous is None or line is None:
        return line is None

    prev_side = side_of_line(previous, line)
    curr_side = side_of_line(center, line)
    return prev_side == 0 or curr_side == 0 or (prev_side < 0 < curr_side) or (curr_side < 0 < prev_side)


def frame_timestamp(start_time: datetime, frame_idx: int, fps: float) -> str:
    ts = start_time + timedelta(seconds=frame_idx / fps)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def sleep_for_realtime_playback(args: argparse.Namespace, start_wall: float, frame_idx: int, fps: float) -> None:
    if not args.realtime_playback:
        return
    target_elapsed = frame_idx / max(fps, 1.0)
    remaining = target_elapsed - (time.monotonic() - start_wall)
    if remaining > 0:
        time.sleep(remaining)


def resolve_classifier_path(path: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path

    fallback = Path("runs/efficientnetb0_mtl/best.pt")
    if checkpoint_path == Path("runs/efficientnetb0_brand/best.pt") and fallback.exists():
        print(f"brand-only checkpoint not found, using existing MTL checkpoint for brand head: {fallback}")
        return fallback

    onnx_fallback = Path("exports/efficientnetb0_brand_opset12.onnx")
    if checkpoint_path == Path("exports/efficientnetb0_brand_fp16.engine") and onnx_fallback.exists():
        print(f"TensorRT brand engine not found, using ONNX fallback: {onnx_fallback}")
        return onnx_fallback

    raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")


def write_metadata(args: argparse.Namespace) -> int:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier = load_brand_classifier(resolve_classifier_path(args.classifier), device, args.classifier_imgsz)

    detector = YOLO(normalize_weights_name(args.detector))
    allowed_classes = {name.strip().lower() for name in args.vehicle_classes.split(",") if name.strip()}
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    fps = video_fps(args.source)
    vid_stride = resolve_vid_stride(args, fps)
    effective_fps = fps / vid_stride
    line = parse_line(args.line)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    emitted_ids: set[int] = set()
    metadata_by_id: dict[int, dict[str, object]] = {}
    color_history_by_id: dict[int, list[str]] = {}
    previous_centers: dict[int, tuple[float, float]] = {}
    video_writer = None
    web_server = None
    web_state = None
    start_wall = time.monotonic()

    if args.view:
        web_server, web_state = start_web_view(args)

    print(
        f"source_fps={fps:.2f} vid_stride={vid_stride} "
        f"effective_detect_fps={effective_fps:.2f} emit_mode={args.emit_mode}"
    )

    results = detector.track(
        source=args.source,
        stream=True,
        persist=True,
        imgsz=args.detector_imgsz,
        conf=args.detector_conf,
        iou=args.detector_iou,
        device=str(device) if device.type == "cpu" else device.index or 0,
        vid_stride=vid_stride,
        verbose=False,
    )

    count = 0
    try:
        with output.open("w", encoding="utf-8") as file:
            for processed_idx, result in enumerate(results):
                frame_idx = processed_idx * vid_stride
                if args.max_frames is not None and processed_idx >= args.max_frames:
                    break

                annotated = result.orig_img.copy() if args.view or args.save_video else None
                boxes = result.boxes
                if boxes is None or len(boxes) == 0 or boxes.id is None:
                    if annotated is not None:
                        cv2.putText(
                            annotated,
                            f"Frame {frame_idx}",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        video_writer = handle_video_output(args, annotated, effective_fps, video_writer, web_state)
                        sleep_for_realtime_playback(args, start_wall, frame_idx, fps)
                    continue

                xyxy = boxes.xyxy.detach().cpu().numpy()
                track_ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
                classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()

                for box, track_id, class_id in zip(xyxy, track_ids, classes):
                    class_name = detector.names.get(class_id, str(class_id)).lower()
                    color_bgr = color_for_track(track_id)
                    x1, y1, x2, y2 = [float(v) for v in box]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    should_emit = args.emit_mode == "every_detection" or track_id not in emitted_ids
                    if allowed_classes and class_name not in allowed_classes:
                        should_emit = False
                    if should_emit and not crossed_line(track_id, center, line, previous_centers):
                        should_emit = False

                    if should_emit:
                        crop = crop_box(result.orig_img, box, args.crop_pad)
                        if crop is not None and min(crop.shape[:2]) >= args.min_crop_size:
                            brand_prediction = classify_brand(crop, classifier)
                            raw_color, raw_color_confidence = estimate_vehicle_color(crop, method=args.color_method)
                            vehicle_color, color_vote_ratio, color_vote_count = update_color_vote(
                                color_history_by_id,
                                track_id,
                                raw_color,
                                args.color_vote_window,
                            )
                            save_color_debug_crop(args, crop, frame_idx, track_id, raw_color)
                            payload = {
                                "timestamp": frame_timestamp(start_time, frame_idx, fps),
                                "vehicle_id": track_id,
                                "brand": brand_prediction["brand"],
                                "color": vehicle_color,
                            }

                            metadata_by_id[track_id] = payload
                            if args.include_confidence:
                                payload["brand_confidence"] = brand_prediction["brand_confidence"]
                                payload["raw_color"] = raw_color
                                payload["raw_color_confidence"] = raw_color_confidence
                                payload["color_vote_ratio"] = color_vote_ratio
                                payload["color_vote_count"] = color_vote_count

                            payload_json = json.dumps(payload, ensure_ascii=False)
                            file.write(payload_json + "\n")
                            file.flush()
                            if args.print_json:
                                print(payload_json, flush=True)
                            emitted_ids.add(track_id)
                            count += 1

                    if annotated is None:
                        continue

                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), color_bgr, 2)
                    vehicle_meta = metadata_by_id.get(track_id)
                    if vehicle_meta:
                        label = f"ID {track_id} {class_name} {vehicle_meta['brand']} {vehicle_meta['color']}"
                    else:
                        label = f"ID {track_id} {class_name}"
                    draw_label(annotated, label, x1i, y1i, color_bgr)

                if annotated is not None:
                    cv2.putText(
                        annotated,
                        f"Frame {frame_idx}",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    video_writer = handle_video_output(args, annotated, effective_fps, video_writer, web_state)
                sleep_for_realtime_playback(args, start_wall, frame_idx, fps)
    finally:
        if video_writer is not None:
            video_writer.release()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()

    return count


def handle_video_output(args: argparse.Namespace, frame, fps: float, video_writer, web_state: MjpegState | None):
    if web_state is not None:
        web_state.update(frame, args.jpeg_quality)

    if args.save_video:
        if video_writer is None:
            h, w = frame.shape[:2]
            output_path = Path(args.video_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_path), fourcc, args.video_fps or fps, (w, h))
            if not video_writer.isOpened():
                raise OSError(f"Could not create output video: {output_path}")
        video_writer.write(frame)
    return video_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO tracking + brand classifier + OpenCV vehicle color metadata.")
    parser.add_argument("--source", default="Test Video.mp4", help="Video path or RTSP URL.")
    parser.add_argument("--detector", default="yolo26n.pt", help="YOLO detector weights.")
    parser.add_argument(
        "--classifier",
        default="exports/efficientnetb0_brand_opset12.onnx",
        help="Brand classifier: .pt, .onnx, or TensorRT .engine.",
    )
    parser.add_argument("--output", default="vehicle_metadata.jsonl", help="One JSON object per vehicle.")
    parser.add_argument("--start-time", default="2026-05-19 10:30:00", help="Video timestamp at frame 0.")
    parser.add_argument("--line", default=None, help="Optional counting line: x1,y1,x2,y2. If omitted, classify first seen ID.")
    parser.add_argument("--vehicle-classes", default="car,truck,bus,motorcycle")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.7)
    parser.add_argument("--detect-fps", type=float, default=30.0, help="Target detection FPS. Default targets 30 FPS.")
    parser.add_argument("--vid-stride", type=int, default=None, help="Manual frame stride. Overrides --detect-fps.")
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--color-method", choices=("kmeans", "hsv"), default="kmeans")
    parser.add_argument(
        "--color-vote-window",
        type=int,
        default=15,
        help="Rolling color-vote window per vehicle_id. Use 0 for all observations.",
    )
    parser.add_argument("--save-color-crops", default=None, help="Optional directory for crop images named with predicted color.")
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--emit-mode",
        choices=("every_detection", "first_seen"),
        default="every_detection",
        help="every_detection classifies/logs each processed detection; first_seen logs once per track_id.",
    )
    parser.add_argument("--include-confidence", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="Print each emitted metadata payload for live streams.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames for a quick test run.")
    parser.add_argument("--view", action="store_true", help="Serve annotated video as a browser link. No local UI window is opened.")
    parser.add_argument("--save-video", action="store_true", help="Save the annotated stream/video to MP4.")
    parser.add_argument("--video-output", default="live_vehicle_metadata_annotated.mp4")
    parser.add_argument("--video-fps", type=float, default=None, help="Override output FPS for live streams.")
    parser.add_argument("--web-host", default="0.0.0.0", help="Host for --view MJPEG server.")
    parser.add_argument("--web-port", type=int, default=8080, help="Port for --view MJPEG server.")
    parser.add_argument("--web-fps", type=float, default=10.0, help="MJPEG serving FPS for browser view.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for browser view.")
    parser.add_argument("--realtime-playback", action="store_true", help="Throttle file/video playback to source FPS for demos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        count = write_metadata(args)
        print(f"saved {count} metadata rows to {args.output}")
    except KeyboardInterrupt:
        print("stopped by user")


if __name__ == "__main__":
    main()
