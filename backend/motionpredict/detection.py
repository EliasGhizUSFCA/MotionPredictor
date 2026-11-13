"""Person detection + pose with an Ultralytics YOLO-pose model.

YOLO11-pose detects the COCO ``person`` class and regresses 17 body keypoints in one pass,
so detection and pose estimation share a single forward pass. No face recognition or any
identity features are computed; keypoints are coarse body joints only.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .schemas import Detection

WEIGHTS_DIR = Path(os.environ.get("MP_WEIGHTS_DIR", Path(__file__).resolve().parents[2] / "models" / "detector"))


class PersonDetector:
    def __init__(self, weights: str | None = None, device: str = "cpu"):
        from ultralytics import YOLO

        weights = weights or os.environ.get("MP_DETECTOR", "yolo11n-pose.pt")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        path = WEIGHTS_DIR / weights
        # Ultralytics downloads official weights by name if the file is not present.
        cwd = os.getcwd()
        try:
            os.chdir(WEIGHTS_DIR)
            self.model = YOLO(str(path) if path.exists() else weights)
        finally:
            os.chdir(cwd)
        self.weights = weights
        self.device = device
        self.is_pose = "pose" in weights
        self.load_time = time.time()

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.35, iou: float = 0.5, imgsz: int = 640) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, conf=conf, iou=iou, imgsz=imgsz, classes=[0], device=self.device, verbose=False
        )[0]
        dets: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kps = None
        if self.is_pose and res.keypoints is not None and res.keypoints.data is not None:
            kps = res.keypoints.data.cpu().numpy()  # (N,17,3)
        for i in range(len(boxes)):
            dets.append(
                Detection(
                    bbox=[float(v) for v in boxes[i]],
                    confidence=float(confs[i]),
                    keypoints=kps[i].round(2).tolist() if kps is not None else None,
                )
            )
        return dets
