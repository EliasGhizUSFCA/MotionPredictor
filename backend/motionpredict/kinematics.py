"""Per-track kinematic state estimation from footpoint observations with real timestamps.

A constant-acceleration Kalman filter smooths the footpoint (bottom-centre of the box, the
point that touches the ground) and yields velocity, acceleration, heading and turn rate.
Units are pixels unless a ground-plane calibration exists, in which case a second filter runs
in meters. Physical momentum is *not* computed: it requires the person's mass.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .forecasting.constants import DT, OBS_LEN
from .kalman import KalmanFilter2D
from .schemas import Kinematics

SPEED_CATEGORIES = [(0.2, "stationary"), (1.0, "slow walk"), (1.8, "walk"), (3.0, "brisk walk / jog"), (math.inf, "running")]


class KinematicTrack:
    def __init__(self, units: str, q: float, r: float, history_s: float = 6.0):
        self.units = units
        self.kf = KalmanFilter2D("ca", q=q, r=r)
        self.history: deque[tuple[float, float, float]] = deque()  # (t, x, y) filtered
        self.history_s = history_s
        self.last_t: float | None = None
        self.scale = 1.0  # units per body-height proxy (px mode: bbox height/1.7)

    def update(self, t: float, z: np.ndarray, r: float | None = None, q: float | None = None) -> None:
        if q is not None:
            self.kf.q = q
        if self.last_t is not None and t < self.last_t:  # never let time run backwards inside a track
            t = self.last_t
        if self.kf.x is None:
            self.kf.init(z, pos_var=r, vel_var=(5 * self.scale) ** 2, acc_var=(5 * self.scale) ** 2)
        else:
            self.kf.predict(t - self.last_t)
            self.kf.update(z, r)
        self.last_t = t
        self.history.append((t, float(self.kf.x[0]), float(self.kf.x[1])))
        while self.history and t - self.history[0][0] > self.history_s:
            self.history.popleft()

    def coast(self, t: float) -> None:
        """Advance prediction without a measurement (track temporarily lost)."""
        if self.kf.x is None or self.last_t is None or t <= self.last_t:
            return
        self.kf.predict(t - self.last_t)
        self.last_t = t

    def state(self) -> Kinematics | None:
        if self.kf.x is None or len(self.history) < 2:
            return None
        x = self.kf.x
        v, a = x[2:4], x[4:6]
        sp = float(np.hypot(*v))
        turn = 0.0
        if sp > 1e-6:
            turn = math.degrees((v[0] * a[1] - v[1] * a[0]) / (sp**2))
        norm_speed = sp / max(self.scale, 1e-6)  # body heights / s in px mode, m/s in meter mode
        if self.units == "m":
            cat = next(name for thr, name in SPEED_CATEGORIES if sp < thr)
            state = "stationary" if sp < 0.2 else "moving"
        else:
            cat = None
            state = "stationary" if norm_speed < 0.15 else "moving"
        if state == "moving" and abs(turn) > 25 and sp > 1e-6:
            state = "turning left" if (turn > 0) == (self.units == "m") else "turning right"
        return Kinematics(
            units=self.units,
            position=[float(x[0]), float(x[1])],
            velocity=[float(v[0]), float(v[1])],
            speed=sp,
            heading_deg=math.degrees(math.atan2(v[1], v[0])),
            acceleration=[float(a[0]), float(a[1])],
            accel_magnitude=float(np.hypot(*a)),
            turn_rate_deg_s=float(np.clip(turn, -720, 720)),
            movement_state=state,
            speed_category=cat,
            position_std=float(math.sqrt(max(self.kf.P[0, 0], self.kf.P[1, 1]))),
        )

    def resampled(self, t_now: float | None = None, n: int = OBS_LEN, dt: float = DT) -> tuple[np.ndarray, int]:
        """Positions on the model's fixed grid ending at the latest observation.

        Returns (n, 2) positions (left padded with the earliest point) and the number of
        valid (non-padded) steps."""
        h = np.asarray(self.history)
        t_end = h[-1, 0] if t_now is None else t_now
        ts = t_end - dt * np.arange(n - 1, -1, -1)
        valid = ts >= h[0, 0] - 1e-6
        xs = np.interp(ts, h[:, 0], h[:, 1])
        ys = np.interp(ts, h[:, 0], h[:, 2])
        out = np.stack([xs, ys], -1)
        nv = int(valid.sum())
        if nv == 0:  # t_now before the first observation: nothing valid, pad with the latest point
            out[:] = h[-1, 1:]
        elif nv < n:
            out[: n - nv] = out[n - nv]
        return out, nv

    def velocity_per_step(self, dt: float = DT) -> np.ndarray:
        return self.kf.x[2:4] * dt
