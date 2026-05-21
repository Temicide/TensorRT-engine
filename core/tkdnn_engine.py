import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from config import CONFIG


log = logging.getLogger("multicam")


class TkDNNDarknetModel:
    """Python-side adapter for a tkDNN/Darknet detector bridge.

    tkDNN is a C++ inference toolkit. Pointing Python at Darknet cfg/weights is
    not enough to run inference inside this process, so this adapter calls a
    configured bridge executable and validates its JSON output.
    """

    backend_name = "tkdnn_darknet"
    supports_batch = False

    def __init__(self, tkdnn_config):
        self.config = dict(tkdnn_config or {})
        self.darknet_dir = Path(self.config.get("darknet_dir", "models/pipeline_2/darknet"))
        self.cfg = Path(self.config.get("cfg", self.darknet_dir / "yolov4-tiny.cfg"))
        self.weights = Path(self.config.get("weights", self.darknet_dir / "yolov4-tiny.weights"))
        self.names = Path(self.config.get("names", self.darknet_dir / "coco.names"))
        self.rt = Path(self.config["rt"]) if self.config.get("rt") else None
        self.bridge_mode = self.config.get("bridge_mode", "image_command")
        self.command = self.config.get("command", "")
        self.timeout_sec = float(self.config.get("timeout_sec", 2.0))
        self.imgsz = int(CONFIG.get("imgsz", 416))

        self._validate_darknet_assets()
        self._validate_rt_asset()
        self._validate_bridge()

        log.info(
            "Loaded tkDNN/Darknet adapter: cfg=%s weights=%s names=%s rt=%s command=%s",
            self.cfg,
            self.weights,
            self.names,
            self.rt,
            self.command,
        )

    def _validate_darknet_assets(self):
        missing = [
            str(path)
            for path in (self.cfg, self.weights, self.names)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "tkDNN Darknet assets are missing:\n"
                "  {}\n\n"
                "Create them with:\n"
                "  cd models/pipeline_2\n"
                "  ./download_yolov4_tiny_darknet.sh".format("\n  ".join(missing))
            )

    def _validate_rt_asset(self):
        if self.rt is not None and not self.rt.is_file():
            raise RuntimeError(
                "tkDNN TensorRT runtime file is missing: {}\n\n"
                "Build it on the Jetson with:\n"
                "  cd /home/ta/tkDNN/build\n"
                "  export TKDNN_MODE=FP16\n"
                "  ./test_yolo4tiny".format(self.rt)
            )

    def _validate_bridge(self):
        if self.bridge_mode != "image_command":
            raise RuntimeError(
                "Unsupported tkDNN bridge_mode={!r}. Currently supported: "
                "'image_command'.".format(self.bridge_mode)
            )

        if not self.command:
            raise RuntimeError(
                "CONFIG['inference_backend'] is set to 'tkdnn_darknet', but "
                "CONFIG['tkdnn']['command'] is empty.\n\n"
                "tkDNN cannot be called directly from this Python server with "
                "only yolov4-tiny.cfg/yolov4-tiny.weights. Build or provide a "
                "small tkDNN executable bridge, then set:\n"
                "  CONFIG['tkdnn']['command'] = '/path/to/tkdnn_json_infer'\n\n"
                "Expected bridge command:\n"
                "  <command> --rt <rt> --cfg <cfg> --weights <weights> --names <names> "
                "--image <jpg> --conf <float> --iou <float>\n\n"
                "Expected stdout JSON:\n"
                "  [{\"class_id\":2,\"class_name\":\"car\",\"confidence\":0.91,"
                "\"bbox_xyxy\":[10,20,120,220]}]\n\n"
                "If you want to keep the current Python TensorRT runtime, set "
                "CONFIG['inference_backend'] = 'tensorrt_engine'."
            )

        resolved = shutil.which(self.command) if not Path(self.command).is_file() else self.command
        if resolved is None:
            raise RuntimeError(
                "tkDNN bridge command not found: {}. Set CONFIG['tkdnn']['command'] "
                "to an executable path.".format(self.command)
            )
        self.command = resolved

    def infer_frame(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode frame for tkDNN bridge")

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(encoded.tobytes())
            image_file.flush()

            cmd = [
                self.command,
            ]
            if self.rt is not None:
                cmd.extend(["--rt", str(self.rt)])
            cmd.extend([
                "--cfg",
                str(self.cfg),
                "--weights",
                str(self.weights),
                "--names",
                str(self.names),
                "--image",
                image_file.name,
                "--conf",
                str(CONFIG["conf_threshold"]),
                "--iou",
                str(CONFIG["iou_threshold"]),
            ])
            completed = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                "tkDNN bridge failed with exit code {}:\n{}".format(
                    completed.returncode,
                    completed.stderr.strip(),
                )
            )

        try:
            detections = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "tkDNN bridge returned invalid JSON: {}\nstdout:\n{}".format(
                    exc,
                    completed.stdout[:1000],
                )
            )

        if not isinstance(detections, list):
            raise RuntimeError("tkDNN bridge JSON must be a list of detections")

        return [self._normalize_detection(detection) for detection in detections]

    @staticmethod
    def _normalize_detection(detection):
        required = ("class_id", "class_name", "confidence", "bbox_xyxy")
        missing = [key for key in required if key not in detection]
        if missing:
            raise RuntimeError("tkDNN detection missing keys: {}".format(missing))

        bbox = detection["bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError("tkDNN detection bbox_xyxy must be a 4-value list")

        return {
            "class_id": int(detection["class_id"]),
            "class_name": str(detection["class_name"]),
            "confidence": round(float(detection["confidence"]), 4),
            "bbox_xyxy": [round(float(value), 1) for value in bbox],
        }
