"""Dataset adapters for ETH/UCY-format pedestrian trajectories.

File format (one row per annotation, whitespace separated):
    frame_id  pedestrian_id  x_meters  y_meters
with consecutive annotated frames 10 frame-ids (0.4 s) apart.

Split policy (no leakage):
* Test  = every file of the held-out scene (leave-one-scene-out, as in the ETH/UCY literature).
* Train/val = remaining scenes, split *temporally* per file: windows whose frames all lie in the
  first ``train_frac`` of the file's frame range go to train, windows fully in the remainder go
  to val; windows straddling the boundary are discarded, so no observation or future position
  is shared between train and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import MAX_NEIGHBORS, NEIGHBOR_RADIUS, OBS_LEN, PRED_LEN

SCENES: dict[str, list[str]] = {
    "eth": ["biwi_eth.txt"],
    "hotel": ["biwi_hotel.txt"],
    "univ": ["students001.txt", "students003.txt"],
    "zara1": ["crowds_zara01.txt"],
    "zara2": ["crowds_zara02.txt"],
}
FRAME_STEP = 10


@dataclass
class WindowSet:
    obs: np.ndarray  # (N, OBS_LEN, 2)
    fut: np.ndarray  # (N, PRED_LEN, 2)
    nbrs: np.ndarray  # (N, MAX_NEIGHBORS, 4)
    nbr_mask: np.ndarray  # (N, MAX_NEIGHBORS)
    scene: np.ndarray  # (N,) scene name

    def __len__(self) -> int:
        return len(self.obs)

    @staticmethod
    def concat(sets: list["WindowSet"]) -> "WindowSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            return WindowSet(
                np.zeros((0, OBS_LEN, 2), np.float32), np.zeros((0, PRED_LEN, 2), np.float32),
                np.zeros((0, MAX_NEIGHBORS, 4), np.float32), np.zeros((0, MAX_NEIGHBORS), np.float32),
                np.zeros((0,), object),
            )
        return WindowSet(*(np.concatenate([getattr(s, f) for s in sets]) for f in ("obs", "fut", "nbrs", "nbr_mask", "scene")))


def load_file(path: Path) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.float64)
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def extract_windows(
    rows: np.ndarray, scene: str, frame_lo: float = -np.inf, frame_hi: float = np.inf, stride: int = 1
) -> WindowSet:
    """Slide (OBS_LEN+PRED_LEN)-long windows over every pedestrian with contiguous annotations."""
    L = OBS_LEN + PRED_LEN
    by_frame: dict[float, np.ndarray] = {}
    for f in np.unique(rows[:, 0]):
        by_frame[f] = rows[rows[:, 0] == f]
    obs_l, fut_l, nb_l, nm_l = [], [], [], []
    for pid in np.unique(rows[:, 1]):
        tr = rows[rows[:, 1] == pid]
        frames = tr[:, 0]
        # split into contiguous runs
        breaks = np.where(np.diff(frames) != FRAME_STEP)[0] + 1
        for run in np.split(np.arange(len(tr)), breaks):
            if len(run) < L:
                continue
            for s in range(0, len(run) - L + 1, stride):
                idx = run[s : s + L]
                fr = frames[idx]
                if fr[0] < frame_lo or fr[-1] >= frame_hi:
                    continue
                xy = tr[idx, 2:4]
                last_f = fr[OBS_LEN - 1]
                nb, nm = neighbours_at(by_frame, last_f, pid, xy[OBS_LEN - 1])
                obs_l.append(xy[:OBS_LEN])
                fut_l.append(xy[OBS_LEN:])
                nb_l.append(nb)
                nm_l.append(nm)
    n = len(obs_l)
    if n == 0:
        return WindowSet.concat([])
    return WindowSet(
        np.asarray(obs_l, np.float32), np.asarray(fut_l, np.float32), np.asarray(nb_l, np.float32),
        np.asarray(nm_l, np.float32), np.asarray([scene] * n, object),
    )


def neighbours_at(by_frame: dict, frame: float, pid: float, ego_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbours present at ``frame`` with per-step (0.4 s) velocity from the previous frame."""
    nb = np.zeros((MAX_NEIGHBORS, 4), np.float32)
    nm = np.zeros((MAX_NEIGHBORS,), np.float32)
    cur = by_frame.get(frame)
    if cur is None:
        return nb, nm
    prev = by_frame.get(frame - FRAME_STEP)
    cand = []
    for r in cur:
        if r[1] == pid:
            continue
        d = float(np.hypot(*(r[2:4] - ego_xy)))
        if d > NEIGHBOR_RADIUS:
            continue
        v = np.zeros(2)
        if prev is not None:
            p = prev[prev[:, 1] == r[1]]
            if len(p):
                v = r[2:4] - p[0, 2:4]
        cand.append((d, r[2], r[3], v[0], v[1]))
    cand.sort()
    for i, c in enumerate(cand[:MAX_NEIGHBORS]):
        nb[i] = c[1:]
        nm[i] = 1
    return nb, nm


def build_splits(raw_dir: Path, test_scene: str, train_frac: float = 0.8, stride: int = 1) -> dict[str, WindowSet]:
    raw_dir = Path(raw_dir)
    train, val, test = [], [], []
    for scene, files in SCENES.items():
        for fn in files:
            path = raw_dir / fn
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run scripts/download_data.sh")
            rows = load_file(path)
            if scene == test_scene:
                test.append(extract_windows(rows, scene, stride=stride))
            else:
                f0, f1 = rows[:, 0].min(), rows[:, 0].max()
                cut = f0 + train_frac * (f1 - f0)
                train.append(extract_windows(rows, scene, frame_hi=cut, stride=stride))
                val.append(extract_windows(rows, scene, frame_lo=cut, stride=stride))
    return {"train": WindowSet.concat(train), "val": WindowSet.concat(val), "test": WindowSet.concat(test)}


def augment(obs, fut, nbrs, rng: np.random.Generator):
    """Random rotation + reflection (reflection covers image-plane y-down handedness)."""
    th = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], np.float32)
    if rng.random() < 0.5:
        R = R @ np.array([[1, 0], [0, -1]], np.float32)
    o = obs[-1].copy()
    obs = (obs - o) @ R.T + o
    fut = (fut - o) @ R.T + o
    nb = nbrs.copy()
    nb[:, :2] = (nbrs[:, :2] - o) @ R.T + o
    nb[:, 2:] = nbrs[:, 2:] @ R.T
    return obs, fut, nb


def left_pad_mask(obs: np.ndarray, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the last ``n_valid`` positions; pad earlier steps with the first valid position."""
    T = obs.shape[0]
    mask = np.zeros(T, np.float32)
    mask[T - n_valid :] = 1
    out = obs.copy()
    out[: T - n_valid] = obs[T - n_valid]
    return out, mask
