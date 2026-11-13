"""Dataset-replay demo mode (explicitly labelled in every payload as ``source='dataset_replay'``).

Replays annotated pedestrian positions from a held-out ETH/UCY scene through the *real*
tracker -> kinematics -> forecaster pipeline. Only the detector is replaced: detections are
synthesised from the annotations, and the "frame" is a synthetic top-down rendering, not
camera footage. A virtual top-down camera provides an exact ground-plane calibration, so the
coordinate system is metric. The true future positions are attached so the UI can compare
predictions against what actually happened.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import GroundCalibration
from .forecasting.data import FRAME_STEP, SCENES, load_file
from .schemas import Detection

PX_PER_M = 40.0
FRAME_DT = 0.4  # seconds per annotated frame


class ReplayDetector:
    """Stands in for the detector: returns detections derived from dataset annotations."""

    weights = "dataset annotations (no detector)"
    is_pose = False

    def __init__(self):
        self.next: list[Detection] = []

    def __call__(self, frame, **_):
        return self.next


class SceneReplay:
    def __init__(self, raw_dir: Path, scene: str, tick_s: float = 0.1, file_index: int = 0, noise_m: float = 0.0, seed: int = 0):
        fn = SCENES[scene][min(file_index, len(SCENES[scene]) - 1)]
        path = Path(raw_dir) / fn
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/download_data.sh to enable demo playback")
        self.rows = load_file(path)
        self.scene = scene
        self.file = fn
        self.tick_s = tick_s
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)
        xy = self.rows[:, 2:4]
        pad = 1.0
        self.xmin, self.ymin = xy.min(0) - pad
        self.xmax, self.ymax = xy.max(0) + pad
        self.W = int((self.xmax - self.xmin) * PX_PER_M)
        self.H = int((self.ymax - self.ymin) * PX_PER_M)
        self.f0, self.f1 = self.rows[:, 0].min(), self.rows[:, 0].max()
        self.t = 0.0
        self.peds: dict[int, np.ndarray] = {}
        for pid in np.unique(self.rows[:, 1]):
            tr = self.rows[self.rows[:, 1] == pid]
            self.peds[int(pid)] = np.c_[(tr[:, 0] - self.f0) / FRAME_STEP * FRAME_DT, tr[:, 2:4]]
        self.duration = (self.f1 - self.f0) / FRAME_STEP * FRAME_DT
        # virtual top-down camera: normalised image -> ground
        self.calibration = GroundCalibration(
            H=np.array([[self.W / PX_PER_M, 0, self.xmin], [0, -self.H / PX_PER_M, self.ymax], [0, 0, 1.0]]),
            image_points=[], world_points=[], rms_error_m=0.0,
            warnings=["Virtual top-down camera for dataset replay (exact calibration)."],
        )

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[(xy[:, 0] - self.xmin) * PX_PER_M, (self.ymax - xy[:, 1]) * PX_PER_M]

    def positions_at(self, t: float) -> dict[int, np.ndarray]:
        out = {}
        for pid, tr in self.peds.items():
            if tr[0, 0] <= t <= tr[-1, 0]:
                i = np.searchsorted(tr[:, 0], t)
                if i == 0:
                    out[pid] = tr[0, 1:]
                    continue
                t0, t1 = tr[i - 1, 0], tr[i, 0]
                if t1 - t0 > FRAME_DT * 1.5:  # gap in annotation: person not visible
                    continue
                a = (t - t0) / max(t1 - t0, 1e-6)
                out[pid] = tr[i - 1, 1:] * (1 - a) + tr[i, 1:] * a
        return out

    def future_of(self, pid: int, t: float, horizon: float = 5.2) -> list[list[float]]:
        tr = self.peds[pid]
        m = (tr[:, 0] > t) & (tr[:, 0] <= t + horizon)
        return [[float(x), float(y), round(float(tt - t), 2)] for tt, x, y in tr[m]]

    def step(self, det: ReplayDetector) -> tuple[np.ndarray, float, dict[int, list[float]]]:
        """Advance one tick; set the detector's detections; return (frame, t, det_box->pid)."""
        self.t += self.tick_s
        if self.t > self.duration:
            self.t = 0.0
        pos = self.positions_at(self.t)
        frame = np.full((self.H, self.W, 3), (24, 20, 18), np.uint8)
        for gx in np.arange(np.ceil(self.xmin), self.xmax, 1.0):
            x = int((gx - self.xmin) * PX_PER_M)
            cv2.line(frame, (x, 0), (x, self.H), (40, 36, 34), 1)
        for gy in np.arange(np.ceil(self.ymin), self.ymax, 1.0):
            y = int((self.ymax - gy) * PX_PER_M)
            cv2.line(frame, (0, y), (self.W, y), (40, 36, 34), 1)
        cv2.putText(frame, "DATASET REPLAY - synthetic top-down view of annotated positions (1 grid = 1 m)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 255), 1, cv2.LINE_AA)
        dets, mapping = [], {}
        for pid, p in pos.items():
            if self.noise_m:
                p = p + self.rng.normal(0, self.noise_m, 2)
            u, v = self.to_px(p)[0]
            box = [u - 8, v - 16, u + 8, v]  # footpoint = bottom centre = annotated position
            dets.append(Detection(bbox=[float(b) for b in box], confidence=0.99))
            mapping[pid] = box
            cv2.circle(frame, (int(u), int(v)), 5, (200, 200, 200), -1)
        det.next = dets
        return frame, self.t, mapping
