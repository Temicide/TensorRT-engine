import atexit
import collections
import concurrent.futures
import json
import logging
import select
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
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
        self.timeout_sec = float(self.config.get("timeout_sec", 15.0))
        self.imgsz = int(CONFIG.get("imgsz", 416))
        self._process = None
        self._process_lock = threading.Lock()
        self._stderr_tail = collections.deque(maxlen=20)
        self._stderr_lock = threading.Lock()

        if self.timeout_sec <= 0:
            raise RuntimeError("CONFIG['tkdnn']['timeout_sec'] must be greater than 0")

        self._validate_darknet_assets()
        self._validate_rt_asset()
        self._validate_bridge()
        atexit.register(self.close)

        log.info(
            "Loaded tkDNN/Darknet adapter: cfg=%s weights=%s names=%s rt=%s command=%s mode=%s timeout=%.1fs",
            self.cfg,
            self.weights,
            self.names,
            self.rt,
            self.command,
            self.bridge_mode,
            self.timeout_sec,
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
        if self.bridge_mode not in ("image_command", "persistent_command"):
            raise RuntimeError(
                "Unsupported tkDNN bridge_mode={!r}. Currently supported: "
                "'image_command' and 'persistent_command'.".format(self.bridge_mode)
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

    def close(self):
        with self._process_lock:
            self._stop_persistent_bridge_locked()

    def infer_frame(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode frame for tkDNN bridge")

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(encoded.tobytes())
            image_file.flush()

            if self.bridge_mode == "persistent_command":
                detections = self._infer_frame_persistent(image_file.name)
            else:
                detections = self._infer_frame_command(image_file.name)

        return [self._normalize_detection(detection) for detection in detections]

    def _build_command(self, image_path=None, server=False):
        cmd = [self.command]
        if self.rt is not None:
            cmd.extend(["--rt", str(self.rt)])
        cmd.extend([
            "--cfg",
            str(self.cfg),
            "--weights",
            str(self.weights),
            "--names",
            str(self.names),
            "--conf",
            str(CONFIG["conf_threshold"]),
            "--iou",
            str(CONFIG["iou_threshold"]),
            "--max-detections",
            str(CONFIG.get("max_detections", 50)),
        ])
        if image_path is not None:
            cmd.extend(["--image", image_path])
        if server:
            cmd.append("--server")
        return cmd

    def _infer_frame_command(self, image_path):
        cmd = self._build_command(image_path=image_path)
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "tkDNN bridge timed out after {:.1f}s. The bundled "
                "tkdnn_json_infer bridge cold-starts Python, PyCUDA, "
                "TensorRT, and the .rt engine for each call on Jetson. "
                "Use bridge_mode='persistent_command' or replace it with a "
                "native bridge for multi-camera use. Command: {}".format(
                    self.timeout_sec,
                    " ".join(shlex.quote(str(part)) for part in cmd),
                )
            ) from None

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

        return detections

    def _infer_frame_persistent(self, image_path):
        request = json.dumps(
            {
                "image": image_path,
                "conf": CONFIG["conf_threshold"],
                "iou": CONFIG["iou_threshold"],
                "max_detections": CONFIG.get("max_detections", 50),
            },
            separators=(",", ":"),
        )

        with self._process_lock:
            self._start_persistent_bridge_locked()
            process = self._process

            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except (IOError, OSError) as exc:
                stderr = self._format_stderr_tail()
                self._stop_persistent_bridge_locked()
                raise RuntimeError(
                    "tkDNN persistent bridge is not writable: {}\nstderr:\n{}".format(
                        exc,
                        stderr,
                    )
                )

            response = self._read_persistent_response_locked()

        if not response.get("ok"):
            raise RuntimeError(
                "tkDNN persistent bridge failed: {}\nstderr:\n{}".format(
                    response.get("error", "unknown error"),
                    self._format_stderr_tail(),
                )
            )

        detections = response.get("detections")
        if not isinstance(detections, list):
            raise RuntimeError("tkDNN persistent bridge JSON missing detections list")
        return detections

    def _start_persistent_bridge_locked(self):
        if self._process is not None and self._process.poll() is None:
            return

        with self._stderr_lock:
            self._stderr_tail.clear()

        cmd = self._build_command(server=True)
        log.info(
            "Starting persistent tkDNN bridge: %s",
            " ".join(shlex.quote(str(part)) for part in cmd),
        )
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
        stderr_thread = threading.Thread(
            target=self._collect_stderr,
            args=(self._process,),
            daemon=True,
            name="tkdnn-bridge-stderr",
        )
        stderr_thread.start()

    def _stop_persistent_bridge_locked(self):
        process = self._process
        self._process = None
        if process is None:
            return

        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

        if process.poll() is not None:
            return

        try:
            process.terminate()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _collect_stderr(self, process):
        try:
            for line in process.stderr:
                line = line.rstrip()
                if line:
                    with self._stderr_lock:
                        self._stderr_tail.append(line)
                    if line == "tkdnn_json_infer server ready":
                        log.info("[tkdnn bridge] %s", line)
        except Exception:
            pass

    def _format_stderr_tail(self):
        with self._stderr_lock:
            lines = list(self._stderr_tail)
        return "\n".join(lines) if lines else "(no stderr captured)"

    def _read_persistent_response_locked(self):
        process = self._process
        deadline = time.time() + self.timeout_sec

        while True:
            if process.poll() is not None:
                stderr = self._format_stderr_tail()
                raise RuntimeError(
                    "tkDNN persistent bridge exited with code {} before response:\n{}".format(
                        process.returncode,
                        stderr,
                    )
                )

            remaining = deadline - time.time()
            if remaining <= 0:
                stderr = self._format_stderr_tail()
                self._stop_persistent_bridge_locked()
                raise RuntimeError(
                    "tkDNN persistent bridge timed out after {:.1f}s. "
                    "This usually means TensorRT engine startup/inference is "
                    "still slower than timeout_sec, or the bridge is stuck. "
                    "Try one camera and increase CONFIG['tkdnn']['timeout_sec'] "
                    "for the first request if stderr shows normal startup.\n"
                    "stderr:\n{}".format(self.timeout_sec, stderr)
                )

            readable, _, _ = select.select([process.stdout], [], [], min(remaining, 0.25))
            if not readable:
                continue

            line = process.stdout.readline()
            if not line:
                continue

            line = line.strip()
            try:
                response = json.loads(line)
            except ValueError:
                with self._stderr_lock:
                    self._stderr_tail.append("stdout: {}".format(line[:500]))
                continue

            if not isinstance(response, dict):
                with self._stderr_lock:
                    self._stderr_tail.append("stdout JSON was not an object: {}".format(line[:500]))
                continue

            return response

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


class TkDNNDarknetModelPool:
    """Pool of persistent tkDNN bridge processes for parallel GPU inference.

    Instead of one bridge process handling all cameras sequentially (which
    leaves the GPU idle during CPU/I/O gaps between requests), this pool
    runs N bridge processes concurrently.  Each camera is assigned a
    dedicated bridge instance so that multiple TensorRT engines submit
    work to the GPU at the same time, keeping utilisation high.

    Each bridge instance internally keeps its own persistent subprocess
    (started lazily on first inference).  Warm-up is staggered so that
    the Jetson Nano does not try to create N CUDA contexts at once.

    If warm-up fails for an instance (e.g. GPU memory exhausted), that
    instance is removed from the pool so it never blocks real inference.
    """

    backend_name = "tkdnn_darknet_pool"
    supports_batch = False

    def __init__(self, tkdnn_config, num_instances):
        self.config = dict(tkdnn_config or {})
        self.num_instances = int(num_instances)
        if self.num_instances < 1:
            raise RuntimeError("num_bridge_instances must be >= 1, got {}".format(self.num_instances))

        self.instances = [TkDNNDarknetModel(tkdnn_config) for _ in range(self.num_instances)]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_instances, thread_name_prefix="tkdnn-infer",
        )
        atexit.register(self.close)

        log.info(
            "tkDNN bridge pool created: %d instance(s) for parallel inference",
            self.num_instances,
        )

    def warmup(self, stagger_sec=5.0, warmup_timeout_sec=120.0):
        """Start all persistent bridges one-by-one with a delay.

        Sends a dummy (black) frame through each bridge to trigger TensorRT
        engine loading.  Staggering avoids N simultaneous CUDA context
        creations on Jetson Nano which can exhaust GPU memory.

        Instances that fail warmup are closed and removed from the pool so
        they never block real inference.  A separate warmup_timeout_sec
        (default 120 s) gives engine startup more time than the normal
        inference timeout.
        """
        import numpy as np
        dummy = np.zeros(
            (int(CONFIG.get("imgsz", 416)), int(CONFIG.get("imgsz", 416)), 3),
            dtype=np.uint8,
        )

        original_timeouts = []
        for inst in self.instances:
            original_timeouts.append(inst.timeout_sec)

        alive = []
        for i, inst in enumerate(self.instances):
            log.info("Warming up bridge instance %d/%d ...", i + 1, self.num_instances)
            inst.timeout_sec = warmup_timeout_sec
            try:
                inst.infer_frame(dummy)
                alive.append(inst)
            except Exception as exc:
                log.warning(
                    "Bridge instance %d/%d warmup failed, removing from pool: %s",
                    i + 1, self.num_instances, exc,
                )
                try:
                    inst.close()
                except Exception:
                    pass
            if i < self.num_instances - 1:
                time.sleep(stagger_sec)

        removed = self.num_instances - len(alive)
        self.instances = alive
        self.num_instances = len(alive)

        if self.num_instances == 0:
            raise RuntimeError(
                "All {} tkDNN bridge instances failed warmup. GPU memory may be "
                "exhausted. Reduce num_bridge_instances in config or free GPU memory.".format(
                    self.num_instances + removed
                )
            )

        if removed > 0:
            log.warning(
                "Removed %d bridge instance(s) that failed warmup. Pool size: %d",
                removed, self.num_instances,
            )

        for inst in self.instances:
            inst.timeout_sec = float(self.config.get("timeout_sec", 60.0))

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_instances, thread_name_prefix="tkdnn-infer",
        )

        log.info("Bridge pool ready: %d healthy instance(s).", self.num_instances)

    def infer_frames_parallel(self, cam_ids, frames):
        """Infer N frames in parallel across the bridge pool.

        Returns a list of detection lists, one per frame, in the same order
        as *cam_ids* / *frames*.
        """
        n = len(cam_ids)
        if n == 0:
            return []

        results = [None] * n

        def _infer_one(idx):
            inst = self.instances[idx % self.num_instances]
            try:
                results[idx] = inst.infer_frame(frames[idx])
            except RuntimeError as exc:
                log.error("[%s] tkdnn_darknet_pool inference failed: %s", cam_ids[idx], exc)
                results[idx] = []
            except Exception as exc:
                log.exception("[%s] tkdnn_darknet_pool inference failed: %s", cam_ids[idx], exc)
                results[idx] = []

        futures = [self._executor.submit(_infer_one, i) for i in range(n)]
        concurrent.futures.wait(futures)
        return results

    def close(self):
        for inst in self.instances:
            inst.close()
        self._executor.shutdown(wait=False)
