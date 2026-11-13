"""Global camera-motion check.

Sparse Lucas-Kanade optical flow on background corners (person boxes masked out). The median
background displacement approximates camera pan/shake. If it exceeds a threshold, apparent
pedestrian motion is contaminated by camera motion and the UI shows a warning. Compensation
is deliberately not applied (a single translation is a poor model of 3-D camera motion).
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(self, work_width: int = 320, threshold_frac: float = 0.004):
        self.work_width = work_width
        self.threshold_frac = threshold_frac
        self.prev: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self.scale = 1.0

    def reset(self) -> None:
        self.prev = None
        self.prev_pts = None

    def update(self, frame_bgr: np.ndarray, boxes: list[list[float]], dt: float) -> dict:
        h, w = frame_bgr.shape[:2]
        self.scale = self.work_width / w
        small = cv2.resize(frame_bgr, (self.work_width, int(h * self.scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (np.array(b) * self.scale).astype(int)
            mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 0
        result = {"available": False, "shift_px": [0.0, 0.0], "shift_px_per_s": 0.0, "camera_moving": False, "tracked_points": 0}
        if self.prev is not None and self.prev_pts is not None and len(self.prev_pts) >= 8 and self.prev.shape == gray.shape:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.prev_pts, None, winSize=(15, 15), maxLevel=2)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 8:
                d = (nxt[ok] - self.prev_pts[ok]).reshape(-1, 2)
                med = np.median(d, 0) / self.scale
                mag = float(np.hypot(*med))
                result = {
                    "available": True,
                    "shift_px": [float(med[0]), float(med[1])],
                    "shift_px_per_s": mag / max(dt, 1e-3),
                    "camera_moving": mag > self.threshold_frac * w,
                    "tracked_points": int(ok.sum()),
                }
        self.prev = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=8, mask=mask)
        return result
