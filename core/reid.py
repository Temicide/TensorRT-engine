import logging

import cv2
import numpy as np

log = logging.getLogger("multicam")


class MobileNetv2ReID:
    FEATURE_DIM = 128

    def __init__(self, model_path=None, imgsz=128, device=0):
        self.imgsz = imgsz
        self._model = None
        self._initialized = False

        if model_path is None:
            log.info("ReID: no model_path specified, using histogram feature extractor")
            self._init_opencv_extractor()
        elif model_path.endswith(".onnx"):
            self._init_onnx(model_path)
        else:
            self._init_opencv_extractor()

    def _init_onnx(self, model_path):
        try:
            self._model = cv2.dnn.readNetFromONNX(model_path)
            self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._initialized = True
            log.info("ReID: loaded ONNX model from %s", model_path)
        except Exception as e:
            log.warning("ReID: failed to load ONNX model %s: %s. Falling back to histogram extractor.", model_path, e)
            self._init_opencv_extractor()

    def _init_opencv_extractor(self):
        self._model = None
        self._initialized = True
        log.info("ReID: using color/histogram feature extractor (no neural network)")

    def extract(self, frame, bboxes):
        if not self._initialized:
            return [None] * len(bboxes)

        features = []
        for bbox in bboxes:
            feat = self._extract_one(frame, bbox)
            features.append(feat)
        return features

    def _extract_one(self, frame, bbox):
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

    def _extract_onnx(self, crop):
        try:
            blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
            self._model.setInput(blob)
            feat = self._model.forward()
            feat = feat.flatten()
            feat = feat / (np.linalg.norm(feat) + 1e-6)
            if len(feat) > self.FEATURE_DIM:
                feat = feat[:self.FEATURE_DIM]
            elif len(feat) < self.FEATURE_DIM:
                pad = np.zeros(self.FEATURE_DIM - len(feat), dtype=np.float32)
                feat = np.concatenate([feat, pad])
            return feat.astype(np.float32)
        except Exception as e:
            log.debug("ReID ONNX extraction failed: %s", e)
            return self._extract_histogram(crop)

    @staticmethod
    def _extract_histogram(crop):
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