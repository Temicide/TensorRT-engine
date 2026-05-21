import numpy as np


class KalmanFilter:
    def __init__(self, bbox: np.ndarray):
        self._motion_mat = np.eye(8, 8)
        self._update_motion_mat()
        self._observation_mat = np.zeros((4, 8))
        self._update_observation_mat()

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        mean = np.zeros((8,))
        mean[:4] = self._bbox_to_z(bbox)
        h = float(bbox[3] - bbox[1])
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
        ]
        self.mean = mean
        self.covariance = np.diag(np.square(std))

    def _update_motion_mat(self):
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

    def _update_observation_mat(self):
        for i in range(4):
            self._observation_mat[i, i] = 1.0

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r])

    @staticmethod
    def z_to_bbox(z: np.ndarray) -> np.ndarray:
        cx, cy, s, r = z
        s = max(s, 1e-6)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    def _current_height(self) -> float:
        h = self.mean[1] * 2.0
        w = self.mean[2] / max(self.mean[3], 1e-6)
        return max(h, 1.0) if h > 0 else max(float(self.mean[3]), 1.0)

    def predict(self):
        h = max(self.mean[1], 1.0) if self.mean[1] > 0 else 1.0
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
        ]
        motion_cov = np.diag(np.square(std_pos + std_vel))

        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + motion_cov
        return self.mean[:4].copy(), self.covariance[:4, :4].copy()

    def update(self, bbox: np.ndarray):
        z = self._bbox_to_z(bbox)
        h = max(float(bbox[3] - bbox[1]), 1.0)
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
        ]
        innovation_cov = np.diag(np.square(std))

        H = self._observation_mat
        S = H @ self.covariance @ H.T + innovation_cov
        K = self.covariance @ H.T @ np.linalg.inv(S)
        y = z - H @ self.mean
        self.mean = self.mean + K @ y
        I_KH = np.eye(8) - K @ H
        self.covariance = (I_KH @ self.covariance @ I_KH.T) + (K @ innovation_cov @ K.T)

    def get_state_bbox(self) -> np.ndarray:
        return self.z_to_bbox(self.mean[:4])