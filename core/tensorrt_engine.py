import threading
import numpy as np
import cv2
import logging
from config import CONFIG
from utils.utils import COCO_NAMES

try:
    import tensorrt as trt
    import pycuda.driver as cuda
except Exception as e:
    trt = None
    cuda = None
    TRT_IMPORT_ERROR = e
else:
    TRT_IMPORT_ERROR = None

log = logging.getLogger("multicam")

# ─────────────────────────── TensorRT model (Jetson GPU) ─────────────────────
class HostDeviceMem:
    """Pair of page-locked CPU memory and GPU device memory for one binding."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem


class YOLOModel:
    """
    TensorRT YOLO inference wrapper for a pre-built .engine file.

    Important:
      - The .engine must be built on the same Jetson / TensorRT / CUDA stack.
      - This class does NOT use ONNX Runtime.
      - Inference runs on the Jetson GPU through TensorRT.
      - Preprocessing, NMS, drawing, and JPEG encoding are still CPU-side.
    """
    def __init__(self, model_path: str):
        if TRT_IMPORT_ERROR is not None or trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT/PyCUDA import failed. This server now requires TensorRT GPU inference.\n"
                f"Import error: {TRT_IMPORT_ERROR}\n\n"
                "Try checking these on Jetson:\n"
                "  python -c 'import tensorrt as trt; print(trt.__version__)'\n"
                "  python -c 'import pycuda.driver as cuda; print(\"pycuda ok\")'\n"
                "If TensorRT works in system Python but not venv, create the venv with:\n"
                "  python3 -m venv --system-site-packages venv\n"
            )

        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.lock = threading.Lock()

        # PyCUDA contexts are thread-local. This server runs inference from
        # camera threads, so create one explicit CUDA context and push/pop it
        # around every TensorRT call. This avoids crashes such as:
        #   free(): double free detected in tcache 2
        cuda.init()
        self.cuda_device = cuda.Device(0)
        self.cuda_ctx = self.cuda_device.make_context()
        self.stream = cuda.Stream()

        log.info("Created explicit CUDA context on device 0")
        log.info(f"Loading TensorRT engine: {model_path}")
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {model_path}. "
                "The .engine may have been built for a different Jetson/TensorRT/CUDA version."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.num_bindings = int(self.engine.num_bindings)
        self.input_indices = []
        self.output_indices = []

        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_indices.append(i)
            else:
                self.output_indices.append(i)

        if len(self.input_indices) != 1:
            raise RuntimeError(f"Expected exactly 1 TensorRT input, found {len(self.input_indices)}")
        if len(self.output_indices) < 1:
            raise RuntimeError("Expected at least 1 TensorRT output.")

        self.input_idx = self.input_indices[0]
        self.output_idx = self.output_indices[0]
        self.input_name = self.engine.get_binding_name(self.input_idx)

        raw_input_shape = tuple(int(x) for x in self.engine.get_binding_shape(self.input_idx))

        # Handle explicit-batch dynamic engines such as [-1, 3, H, W].
        if any(dim < 0 for dim in raw_input_shape):
            self.imgsz = int(CONFIG["imgsz"])
            self.input_shape = (1, 3, self.imgsz, self.imgsz)
            self.context.set_binding_shape(self.input_idx, self.input_shape)
        else:
            self.input_shape = raw_input_shape
            # Supports NCHW. For YOLO exports this is normally [1, 3, imgsz, imgsz].
            if len(self.input_shape) != 4:
                raise RuntimeError(f"Expected NCHW input shape, got {self.input_shape}")
            self.imgsz = int(self.input_shape[2])

        self.bindings = [None] * self.num_bindings
        self.host_device = {}
        self.binding_shapes = {}

        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(int(x) for x in self.context.get_binding_shape(i))
            if any(dim < 0 for dim in shape):
                shape = tuple(int(x) for x in self.engine.get_binding_shape(i))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Binding {name} still has dynamic shape {shape}. Rebuild engine with fixed shape or set profile.")

            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[i] = int(device_mem)
            self.host_device[i] = HostDeviceMem(host_mem, device_mem)
            self.binding_shapes[i] = shape

            role = "input" if self.engine.binding_is_input(i) else "output"
            log.info(f"TensorRT binding {i}: {role} name={name} shape={shape} dtype={dtype}")

        out_shape = self.binding_shapes[self.output_idx]
        self.output_shape = out_shape

        # Detect YOLOv5 [1,N,85] vs YOLOv8 [1,84,N].
        self.is_v5 = (len(out_shape) == 3 and out_shape[2] in (85, 80 + 5))

        self._warmup()

        # Pop the context created in __init__. It will be pushed again inside
        # _execute() from whichever camera thread is running inference.
        self.cuda_ctx.pop()

        log.info(
            f"TensorRT engine loaded on Jetson GPU: {model_path} | "
            f"input={self.input_shape} | output={self.output_shape} | imgsz={self.imgsz} | v5={self.is_v5}"
        )

    def _execute(self, inp):
        # The CUDA context must be current in the thread that touches CUDA.
        self.cuda_ctx.push()
        try:
            inp = np.ascontiguousarray(inp.astype(np.float32, copy=False))
            input_mem = self.host_device[self.input_idx]
            if inp.size != input_mem.host.size:
                raise RuntimeError("Input size mismatch: got {}, engine expects {}".format(inp.shape, self.input_shape))

            np.copyto(input_mem.host, inp.ravel())
            cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)

            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 failed.")

            outputs = []
            for out_idx in self.output_indices:
                out_mem = self.host_device[out_idx]
                cuda.memcpy_dtoh_async(out_mem.host, out_mem.device, self.stream)
                outputs.append((out_idx, out_mem.host))

            self.stream.synchronize()

            reshaped = []
            for out_idx, host in outputs:
                shape = tuple(int(x) for x in self.context.get_binding_shape(out_idx))
                if any(dim < 0 for dim in shape):
                    shape = self.binding_shapes[out_idx]
                reshaped.append(np.array(host).reshape(shape).copy())
            return reshaped
        finally:
            self.cuda_ctx.pop()
    
    def _ensure_binding_capacity(self, binding_idx):
        shape = tuple(int(x) for x in self.context.get_binding_shape(binding_idx))
        dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
        size = int(trt.volume(shape))

        current = self.host_device.get(binding_idx)
        if current is not None and current.host.size >= size:
            self.binding_shapes[binding_idx] = shape
            return

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        self.host_device[binding_idx] = HostDeviceMem(host_mem, device_mem)
        self.bindings[binding_idx] = int(device_mem)
        self.binding_shapes[binding_idx] = shape

    def _warmup(self):
        dummy = np.zeros(self.input_shape, dtype=np.float32)
        with self.lock:
            _ = self._execute(dummy)
        log.info("TensorRT warmup OK")

    def infer_raw(self, inp: np.ndarray):
        """Run TensorRT on pre-built NCHW float32 input. Returns raw output list."""
        with self.lock:
            return self._execute(inp)

    def decode_output(self, out: np.ndarray, orig_hw: tuple) -> list:
        """Decode raw TensorRT output for ONE image into detection dicts."""
        h0, w0 = orig_hw
        if self.is_v5:
            preds = out    # [N, 85]
            mask  = preds[:, 4] * preds[:, 5:].max(axis=1) > CONFIG["conf_threshold"]
            preds = preds[mask]
            if len(preds) == 0:
                return []
            boxes     = preds[:, :4]
            obj       = preds[:, 4]
            cls_probs = preds[:, 5:]
            class_ids = cls_probs.argmax(axis=1)
            scores    = obj * cls_probs[np.arange(len(preds)), class_ids]
        else:
            if len(out.shape) == 3 and out.shape[1] < out.shape[2]:
                preds = out[0].T if out.ndim == 3 else out.T
            else:
                preds = out[0] if out.ndim == 3 else out
            scores_all = preds[:, 4:]
            class_ids  = scores_all.argmax(axis=1)
            scores     = scores_all[np.arange(len(preds)), class_ids]
            mask       = scores > CONFIG["conf_threshold"]
            preds, class_ids, scores = preds[mask], class_ids[mask], scores[mask]
            if len(preds) == 0:
                return []
            boxes = preds[:, :4]

        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * sx
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * sy

        keep = cpu_nms(np.stack([x1,y1,x2,y2], axis=1), scores, CONFIG["iou_threshold"])
        results = []
        for i in keep:
            cid = int(class_ids[i])
            results.append({
                "class_id":   cid,
                "class_name": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
                "confidence": round(float(scores[i]), 4),
                "bbox_xyxy":  [round(float(x1[i]),1), round(float(y1[i]),1),
                               round(float(x2[i]),1), round(float(y2[i]),1)],
            })
        return results

    def infer_batch(self, batch_chw: np.ndarray, orig_shapes: list) -> list:
        """
        Run TensorRT on a batched [N,3,H,W] float32 tensor.
        orig_shapes: list of (h, w) tuples for each image.
        Returns list of N detection-lists.
        """
        with self.lock:
            n = batch_chw.shape[0]
            batch_chw = np.ascontiguousarray(batch_chw.astype(np.float32, copy=False))

            self.cuda_ctx.push()
            try:
                raw_in_shape = tuple(int(x) for x in self.engine.get_binding_shape(self.input_idx))
                if any(dim < 0 for dim in raw_in_shape):
                    self.context.set_binding_shape(self.input_idx, (n, 3, self.imgsz, self.imgsz))

                self._ensure_binding_capacity(self.input_idx)
                for out_idx in self.output_indices:
                    self._ensure_binding_capacity(out_idx)

                inp_mem = self.host_device[self.input_idx]
                input_size = batch_chw.size
                np.copyto(inp_mem.host[:input_size], batch_chw.ravel())
                cuda.memcpy_htod_async(inp_mem.device, inp_mem.host[:input_size], self.stream)

                ok = self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream.handle,
                )
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v2 (batch) failed.")

                outputs = []
                for out_idx in self.output_indices:
                    out_mem = self.host_device[out_idx]
                    out_shape = self.binding_shapes[out_idx]
                    out_size = int(trt.volume(out_shape))
                    cuda.memcpy_dtoh_async(out_mem.host, out_mem.device, self.stream)
                    outputs.append((out_idx, out_mem.host[:out_size].copy(), out_shape))

                self.stream.synchronize()
            finally:
                self.cuda_ctx.pop()

            raw_outputs = []
            for _, host, shape in outputs:
                raw_outputs.append(host.reshape(shape).copy())

            raw = raw_outputs[0]

            results = []
            for i in range(n):
                results.append(self.decode_output(raw[i], orig_shapes[i]))
            return results

def cpu_nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        ix1 = np.maximum(boxes[i,0], boxes[rest,0])
        iy1 = np.maximum(boxes[i,1], boxes[rest,1])
        ix2 = np.minimum(boxes[i,2], boxes[rest,2])
        iy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep
