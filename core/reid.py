import logging

import cv2
import numpy as np

log = logging.getLogger("multicam")


class MobileNetv2ReID:
    FEATURE_DIM = 128

    def __init__(self, model_path: str | None = None, imgsz: int = 128, device: int = 0):
        self.imgsz = imgsz
        self._model = None
        self._initialized = False

        if model_path is None:
            log.info("ReID: no model_path specified, using OpenCV MobileNetv2 feature extractor")
            self._init_opencv_extractor()
        elif model_path.endswith(".onnx"):
            self._init_onnx(model_path)
        else:
            self._init_opencv_extractor()

    def _init_onnx(self, model_path: str):
        try:
            self._model = cv2.dnn.readNetFromONNX(model_path)
            self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._initialized = True
            log.info(f"ReID: loaded ONNX model from {model_path}")
        except Exception as e:
            log.warning(f"ReID: failed to load ONNX model {model_path}: {e}. Falling back to OpenCV extractor.")
            self._init_opencv_extractor()

    def _init_opencv_extractor(self):
        proto_txt = (
            "536 128\n"
            "0 conv 32 3 3 1 1\n"
            "1 relu\n"
            "2 conv 32 1 1 1 0\n"
            "3 relu\n"
            "4 conv 32 3 3 1 1\n"
            "5 relu\n"
            "6 conv 128 1 1 1 0\n"
            "7 relu\n"
            "8 conv 128 1 1 1 0\n"
            "9 relu\n"
            "10 conv 128 3 3 1 1\n"
            "11 relu\n"
        )
        self._model = None
        self._initialized = True
        log.info("ReID: using color/histogram feature extractor (no neural network)")

    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> list[np.ndarray | None]:
        if not self._initialized:
            return [None] * len(bboxes)

        features = []
        for bbox in bboxes:
            feat = self._extract_one(frame, bbox)
            features.append(feat)
        return features

    def _extract_one(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None

        crop = frame[y1:y2, x1:x2]
        if self._model is not None:
            return self._extract_onnx(crop)
        return self._extract_histogram(crop)

    def _extract_onnx(self, crop: np.ndarray) -> np.ndarray:
        try:
            blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
            self._model.setInput(blob)
            feat = self._model.forward()
            feat = feat.flatten()
            feat = feat / (np.linalg.norm(feat) + 1e-6)
            if len(feat) > self.FEATURE_DIM:
                feat = feat[: self.FEATURE_DIM]
            elif len(feat) < self.FEATURE_DIM:
                pad = np.zeros(self.FEATURE_DIM - len(feat), dtype=np.float32)
                feat = np.concatenate([feat, pad])
            return feat.astype(np.float32)
        except Exception as e:
            log.debug(f"ReID ONNX extraction failed: {e}")
            return self._extract_histogram(crop)

    @staticmethod
    def _extract_histogram(crop: np.ndarray) -> np.ndarray:
        try:
            small = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            hist_grad = cv2.calcHist(
                [grad_x.astype(np.uint8)], [0], None, [32], [0, 256]
            ).flatten()
            feat = np.concatenate([hist_h, hist_s, hist_v, hist_grad]).astype(np.float32)
            feat = feat / (np.linalg.norm(feat) + 1e-6)
            target_dim = MobileNetv2ReID.FEATURE_DIM
            if len(feat) >= target_dim:
                return feat[:target_dim]
            return np.concatenate([feat, np.zeros(target_dim - len(feat), dtype=np.float32)])
        except Exception:
            zero = np.zeros(MobileNetv2ReID.FEATURE_DIM, dtype=np.float32)
            zero[0] = 1e-6
            return zero