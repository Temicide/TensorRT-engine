import numpy as np
from scipy.optimize import linear_sum_assignment

from core.kalman import KalmanFilter


def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    xx1 = max(bbox_a[0], bbox_b[0])
    yy1 = max(bbox_a[1], bbox_b[1])
    xx2 = min(bbox_a[2], bbox_b[2])
    yy2 = min(bbox_a[3], bbox_b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h
    area_a = max(0.0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(0.0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    union = area_a + area_b - intersection
    return intersection / max(union, 1e-6)


def iou_cost_matrix(predicted_bboxes: np.ndarray, det_bboxes: np.ndarray) -> np.ndarray:
    n_tracks = len(predicted_bboxes)
    n_dets = len(det_bboxes)
    if n_tracks == 0 or n_dets == 0:
        return np.full((n_tracks, n_dets), 1e5)
    cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
    for t in range(n_tracks):
        for d in range(n_dets):
            cost[t, d] = 1.0 - _iou(predicted_bboxes[t], det_bboxes[d])
    return cost


def appearance_cost_matrix(track_features: list[list[np.ndarray]], det_features: list[np.ndarray | None]) -> np.ndarray:
    n_tracks = len(track_features)
    n_dets = len(det_features)
    cost = np.full((n_tracks, n_dets), 1.0, dtype=np.float32)
    for t in range(n_tracks):
        if not track_features[t]:
            continue
        gallery = np.mean(track_features[t], axis=0)
        gallery = gallery / (np.linalg.norm(gallery) + 1e-6)
        for d in range(n_dets):
            if det_features[d] is None:
                continue
            probe = det_features[d] / (np.linalg.norm(det_features[d]) + 1e-6)
            cost[t, d] = 1.0 - float(np.dot(gallery, probe))
    return cost


class Track:
    _next_id = 1

    def __init__(self, bbox: np.ndarray, feature: np.ndarray | None = None):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter(bbox)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.features: list[np.ndarray] = []
        if feature is not None:
            self.features.append(feature)
        self._update_mean_bbox()

    def _update_mean_bbox(self):
        self._curr_bbox = self.kf.get_state_bbox()

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self._update_mean_bbox()

    def update(self, bbox: np.ndarray, feature: np.ndarray | None = None):
        self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0
        if feature is not None:
            self.features.append(feature)
        self._update_mean_bbox()

    def get_state_bbox(self) -> np.ndarray:
        return self._curr_bbox.copy()

    def mark_missed(self):
        pass


class DeepSORTTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        gating_threshold: float = 9.4877,
        gating_only_position: bool = False,
        appearance_weight: float = 0.5,
        max_features: int = 100,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.max_features = max_features
        self.tracks: list[Track] = []
        self.frame_count = 0

    def _match_iou(self, bboxes: np.ndarray, iou_threshold: float):
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(bboxes)), np.array([], dtype=int)
        if len(bboxes) == 0:
            return np.empty((0, 2), dtype=int), np.array([], dtype=int), np.arange(len(self.tracks))

        predicted_bboxes = np.array([t.get_state_bbox() for t in self.tracks])
        iou_cost = iou_cost_matrix(predicted_bboxes, bboxes)

        high_cost_mask = iou_cost > (1.0 - iou_threshold)
        cost_matrix = iou_cost.copy()
        cost_matrix[high_cost_mask] = 1e5

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(len(bboxes)))
        unmatched_tracks = set(range(len(self.tracks)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 1e4:
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_tracks.discard(r)

        matched_arr = np.array(matched, dtype=int).reshape(-1, 2) if matched else np.empty((0, 2), dtype=int)
        return matched_arr, np.array(sorted(unmatched_dets), dtype=int), np.array(sorted(unmatched_tracks), dtype=int)

    def _match_deep(self, bboxes: np.ndarray, features: list[np.ndarray | None]):
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(bboxes)), np.array([], dtype=int)
        if len(bboxes) == 0:
            return np.empty((0, 2), dtype=int), np.array([], dtype=int), np.arange(len(self.tracks))

        track_feature_galleries = [t.features for t in self.tracks]
        predicted_bboxes = np.array([t.get_state_bbox() for t in self.tracks])

        iou_cost = iou_cost_matrix(predicted_bboxes, bboxes)
        app_cost = appearance_cost_matrix(track_feature_galleries, features)

        has_features = any(f is not None for f in features) and any(len(tf) > 0 for tf in track_feature_galleries)
        if has_features:
            gate_mask = (iou_cost > (1.0 - self.iou_threshold))
            cost_matrix = (1 - self.appearance_weight) * iou_cost + self.appearance_weight * app_cost
            cost_matrix[gate_mask] = 1e5
            high_app_mask = app_cost > 0.7
            cost_matrix[high_app_mask] = 1e5
        else:
            cost_matrix = iou_cost.copy()
            cost_matrix[iou_cost > (1.0 - self.iou_threshold)] = 1e5

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(len(bboxes)))
        unmatched_tracks = set(range(len(self.tracks)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 1e4:
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_tracks.discard(r)

        matched_arr = np.array(matched, dtype=int).reshape(-1, 2) if matched else np.empty((0, 2), dtype=int)
        return matched_arr, np.array(sorted(unmatched_dets), dtype=int), np.array(sorted(unmatched_tracks), dtype=int)

    def update(self, bboxes: np.ndarray, features: list[np.ndarray | None] | None = None):
        self.frame_count += 1

        for track in self.tracks:
            track.predict()

        if len(bboxes) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._collect_tracks()

        if features is None:
            features = [None] * len(bboxes)

        matched, unmatched_dets, unmatched_tracks = self._match_deep(bboxes, features)

        unmatched_tracks_set = set(unmatched_tracks.tolist()) if len(unmatched_tracks) > 0 else set()
        high_iou_tracks = []
        high_iou_dets = []
        for t_idx in unmatched_tracks_set:
            if self.tracks[t_idx].time_since_update == 1:
                high_iou_tracks.append(t_idx)
        if high_iou_tracks and len(unmatched_dets) > 0:
            second_matched, second_unmatched_dets, _ = self._match_iou_second(
                bboxes, np.array(unmatched_dets), np.array(high_iou_tracks)
            )
            if len(second_matched) > 0:
                matched = np.vstack([matched, second_matched]) if len(matched) > 0 else second_matched
                matched_det_indices = set(int(m[1]) for m in second_matched)
                unmatched_dets = np.array([d for d in unmatched_dets if d not in matched_det_indices], dtype=int)

        for match in matched:
            t_idx, d_idx = int(match[0]), int(match[1])
            self.tracks[t_idx].update(bboxes[d_idx], features[d_idx])

        for t_idx in unmatched_tracks:
            if isinstance(t_idx, (int, np.integer)):
                self.tracks[t_idx].mark_missed()

        for d_idx in unmatched_dets:
            if isinstance(d_idx, (int, np.integer)):
                self.tracks.append(Track(bboxes[d_idx], features[d_idx]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._collect_tracks()

    def _match_iou_second(self, bboxes: np.ndarray, det_indices: np.ndarray, track_indices: np.ndarray):
        track_bboxes = np.array([self.tracks[t].get_state_bbox() for t in track_indices])
        det_bboxes = bboxes[det_indices]
        iou_cost = iou_cost_matrix(track_bboxes, det_bboxes)
        high_cost_mask = iou_cost > (1.0 - self.iou_threshold)
        cost_matrix = iou_cost.copy()
        cost_matrix[high_cost_mask] = 1e5

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        matched_dets = set()
        matched_tracks = set()

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 1e4:
                continue
            real_t = int(track_indices[r])
            real_d = int(det_indices[c])
            matched.append((real_t, real_d))
            matched_dets.add(real_d)
            matched_tracks.add(real_t)

        matched_arr = np.array(matched, dtype=int).reshape(-1, 2) if matched else np.empty((0, 2), dtype=int)
        remaining_dets = np.array([d for d in det_indices if d not in matched_dets], dtype=int)
        remaining_tracks = np.array([t for t in track_indices if t not in matched_tracks], dtype=int)
        return matched_arr, remaining_dets, remaining_tracks

    def _collect_tracks(self) -> list[dict]:
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append({
                    "track_id": track.track_id,
                    "bbox": track.get_state_bbox(),
                    "hits": track.hits,
                    "age": track.age,
                    "time_since_update": track.time_since_update,
                })
        return results