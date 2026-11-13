"""ByteTrack-style multi-object tracker (Zhang et al., ECCV 2022), re-implemented with real timestamps.

Why a re-implementation instead of a black box: the UI needs lifecycle states, per-track
confidences and variable-dt motion prediction, and the logic must be unit-testable without a
detector. Association follows ByteTrack:

1. Predict every tracked + lost track to the frame timestamp (constant-velocity box KF).
2. Match high-score detections to tracked+lost tracks by IoU (Hungarian).
3. Match the *low*-score detections to the remaining *tracked* tracks (recovers occluded people).
4. Match remaining high-score detections to tentative tracks; unmatched tentative tracks die.
5. Unmatched tracks become ``lost`` and are removed after ``track_buffer_s`` seconds.
6. Unmatched high-score detections above ``new_track_thresh`` start tentative tracks.

IDs are small session-local integers. They are pseudonymous, carry no appearance features and
are cleared on ``reset()`` (session stop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    mr = {i for i, _ in matches}
    mc = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class BoxKF:
    """Constant-velocity Kalman filter on [cx, cy, w, h] with velocities in units/second."""

    def __init__(self, box: np.ndarray):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        s = max(h, 1.0)
        self.P = np.diag([(0.1 * s) ** 2] * 4 + [(1.0 * s) ** 2] * 4)

    def predict(self, dt: float) -> None:
        dt = max(dt, 1e-3)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        s = max(self.x[3], 1.0)
        qa = (0.6 * s) ** 2  # accel noise (units/s^2)^2 scaled by box height
        G = np.zeros((8, 4))
        G[:4] = np.eye(4) * 0.5 * dt**2
        G[4:] = np.eye(4) * dt
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        self.P = F @ self.P @ F.T + G @ G.T * qa

    def update(self, box: np.ndarray) -> None:
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        s = max(z[3], 1.0)
        R = np.diag([(0.05 * s) ** 2, (0.05 * s) ** 2, (0.05 * s) ** 2, (0.05 * s) ** 2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    track_id: int
    kf: BoxKF
    start_t: float
    last_t: float
    det_conf: float
    keypoints: list | None = None
    state: str = "tentative"
    hits: int = 1
    consecutive_hits: int = 1
    last_det_box: np.ndarray | None = None
    matched_this_frame: bool = True
    meta: dict = field(default_factory=dict)
    kp_t: float | None = None
    kp_box: np.ndarray | None = None

    def current_keypoints(self, now: float, max_age_s: float = 0.6) -> list | None:
        """Latest keypoints, translated with the box if they come from an earlier pose pass."""
        if self.keypoints is None or self.kp_t is None or now - self.kp_t > max_age_s or self.last_det_box is None:
            return None
        if self.kp_box is None:
            return self.keypoints
        d = (self.last_det_box[:2] + self.last_det_box[2:]) / 2 - (self.kp_box[:2] + self.kp_box[2:]) / 2
        return [[k[0] + float(d[0]), k[1] + float(d[1]), k[2]] for k in self.keypoints]

    def confidence(self, now: float, buffer_s: float) -> float:
        """Heuristic tracker confidence in [0,1]: detection score x maturity x recency."""
        maturity = min(1.0, self.hits / 5.0)
        recency = math.exp(-max(0.0, now - self.last_t) / max(buffer_s, 1e-3))
        return float(self.det_conf * (0.5 + 0.5 * maturity) * recency)


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer_s: float = 1.0,
        min_hits: int = 2,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer_s = track_buffer_s
        self.min_hits = min_hits
        self.reset()

    def reset(self, keep_ids: bool = False) -> None:
        """Drop every track. ``keep_ids=True`` continues the id counter, so ids are never reused
        within one run (e.g. after a camera cut or a timestamp discontinuity)."""
        self.tracks: list[Track] = []
        if not keep_ids or not hasattr(self, "next_id"):
            self.next_id = 1
        self.last_t: float | None = None
        self.removed_ids: list[int] = []
        self.frame_count = 0

    def update(self, dets: list[Detection], t: float) -> dict:
        """Returns a summary dict of association statistics for diagnostics."""
        if self.last_t is not None and t < self.last_t - 0.05:  # time went backwards: old tracks are meaningless
            self.reset(keep_ids=True)
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 1e-3)
        self.last_t = t
        self.frame_count += 1
        self.removed_ids = []
        for tr in self.tracks:
            tr.kf.predict(dt)
            tr.matched_this_frame = False

        boxes = np.array([d.bbox for d in dets], float).reshape(-1, 4)
        scores = np.array([d.confidence for d in dets], float)
        hi = np.where(scores >= self.high_thresh)[0]
        lo = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]

        active = [tr for tr in self.tracks if tr.state in ("confirmed", "lost")]
        tentative = [tr for tr in self.tracks if tr.state == "tentative"]

        # 1st association: high dets vs confirmed + lost (IoU fused with detection score as in ByteTrack)
        tb = np.array([tr.kf.box for tr in active]).reshape(-1, 4)
        iou = iou_matrix(tb, boxes[hi])
        cost = 1 - iou * scores[hi][None, :] if len(hi) else 1 - iou
        m1, u_tr, u_hi = linear_assignment(cost, self.match_thresh)
        for i, j in m1:
            self._apply(active[i], dets[hi[j]], t)

        # 2nd association: low dets vs remaining *tracked* (not lost) tracks
        rem = [active[i] for i in u_tr if active[i].state == "confirmed"]
        tb2 = np.array([tr.kf.box for tr in rem]).reshape(-1, 4)
        m2, u_tr2, _ = linear_assignment(1 - iou_matrix(tb2, boxes[lo]), 0.5)
        for i, j in m2:
            self._apply(rem[i], dets[lo[j]], t)

        # 3rd: remaining high dets vs tentative tracks
        hi_left = [hi[j] for j in u_hi]
        tb3 = np.array([tr.kf.box for tr in tentative]).reshape(-1, 4)
        m3, u_tent, u_hi3 = linear_assignment(1 - iou_matrix(tb3, boxes[hi_left]), 0.7)
        for i, j in m3:
            self._apply(tentative[i], dets[hi_left[j]], t)

        # lifecycle updates
        n_lost_new = 0
        for tr in self.tracks:
            if tr.matched_this_frame:
                if tr.state == "tentative" and (tr.consecutive_hits >= self.min_hits or self.frame_count == 1):
                    tr.state = "confirmed"
                elif tr.state == "lost":
                    tr.state = "confirmed"
            else:
                tr.consecutive_hits = 0
                if tr.state == "confirmed":
                    tr.state = "lost"
                    n_lost_new += 1
        keep = []
        for tr in self.tracks:
            dead = (tr.state == "tentative" and not tr.matched_this_frame) or (
                tr.state == "lost" and t - tr.last_t > self.track_buffer_s
            )
            if dead:
                self.removed_ids.append(tr.track_id)
            else:
                keep.append(tr)
        self.tracks = keep

        # new tracks
        n_new = 0
        for j in u_hi3:
            d = dets[hi_left[j]]
            if d.confidence < self.new_track_thresh:
                continue
            tr = Track(self.next_id, BoxKF(np.array(d.bbox)), t, t, d.confidence, d.keypoints)
            tr.last_det_box = np.array(d.bbox)
            if d.keypoints is not None:
                tr.kp_t, tr.kp_box = t, np.array(d.bbox)
            if self.frame_count == 1:
                tr.state = "confirmed"
            self.next_id += 1
            self.tracks.append(tr)
            n_new += 1

        return {
            "high_detections": int(len(hi)),
            "low_detections": int(len(lo)),
            "matched_high": len(m1),
            "recovered_low": len(m2),
            "matched_tentative": len(m3),
            "new_tracks": n_new,
            "newly_lost": n_lost_new,
            "removed": len(self.removed_ids),
            "active": sum(tr.state == "confirmed" for tr in self.tracks),
            "lost": sum(tr.state == "lost" for tr in self.tracks),
            "tentative": sum(tr.state == "tentative" for tr in self.tracks),
        }

    def _apply(self, tr: Track, d: Detection, t: float) -> None:
        tr.kf.update(np.array(d.bbox))
        tr.last_t = t
        tr.det_conf = d.confidence
        if d.keypoints is not None:  # pose may run only on some frames; keep the last set otherwise
            tr.keypoints = d.keypoints
            tr.kp_t = t
            tr.kp_box = np.array(d.bbox)
        tr.hits += 1
        tr.consecutive_hits += 1
        tr.matched_this_frame = True
        tr.last_det_box = np.array(d.bbox)
