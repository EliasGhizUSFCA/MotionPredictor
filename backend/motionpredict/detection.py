"""Person detection (YOLO11 detect) and optional pose estimation (YOLO11-pose).

Detection uses the plain COCO detector because it is markedly more reliable on small/distant
people than the pose model at the same input size (measured on OpenCV's ``vtest.avi``: 3-6 vs
0-1 people per frame at 640 px). When pose is enabled, the pose model runs as a separate stage
and its keypoints are attached to detector boxes by IoU. No face recognition or identity
features are computed; keypoints are 17 coarse COCO body joints.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .schemas import Detection

WEIGHTS_DIR = Path(os.environ.get("MP_WEIGHTS_DIR", Path(__file__).resolve().parents[2] / "models" / "detector"))


def _load(weights: str):
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / weights
    cwd = os.getcwd()
    try:  # Ultralytics downloads official weights by name into the cwd if missing
        os.chdir(WEIGHTS_DIR)
        return YOLO(str(path) if path.exists() else weights)
    finally:
        os.chdir(cwd)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class PersonDetector:
    supports_pose = True

    def __init__(self, weights: str | None = None, pose_weights: str | None = None, device: str = "cpu"):
        self.weights = weights or os.environ.get("MP_DETECTOR", "yolo11n.pt")
        self.pose_weights = pose_weights or os.environ.get("MP_POSE_MODEL", "yolo11n-pose.pt")
        self.model = _load(self.weights)
        self._pose = None
        self.device = device
        self.is_pose = True
        self.load_time = time.time()

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.35, iou: float = 0.5, imgsz: int = 640) -> list[Detection]:
        res = self.model.predict(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, classes=[0], device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [Detection(bbox=[float(v) for v in boxes[i]], confidence=float(confs[i])) for i in range(len(boxes))]

    def pose(self, frame_bgr: np.ndarray, dets: list[Detection], imgsz: int = 640, min_iou: float = 0.4) -> int:
        """Attach keypoints to detections in place; returns how many people received keypoints."""
        if not dets:
            return 0
        if self._pose is None:
            self._pose = _load(self.pose_weights)
        res = self._pose.predict(frame_bgr, conf=0.15, imgsz=max(imgsz, 960), classes=[0], device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return 0
        pb = res.boxes.xyxy.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()
        used, n = set(), 0
        for d in dets:
            best, bi = min_iou, None
            for j in range(len(pb)):
                if j not in used:
                    v = _iou(d.bbox, pb[j])
                    if v > best:
                        best, bi = v, j
            if bi is not None:
                used.add(bi)
                d.keypoints = kps[bi].round(2).tolist()
                n += 1
        return n
