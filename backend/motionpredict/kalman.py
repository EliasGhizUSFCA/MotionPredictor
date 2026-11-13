"""Small, dependency-free Kalman filters with variable time steps.

State conventions
* CV (constant velocity):     x = [px, py, vx, vy]
* CA (constant acceleration): x = [px, py, vx, vy, ax, ay]

Process noise uses the continuous white-noise model on the highest derivative
(white-noise acceleration for CV, white-noise jerk for CA), so noise scales correctly
with irregular frame intervals measured from real timestamps.
"""

from __future__ import annotations

import numpy as np


def _cv_mats(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    g = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
    Q = np.zeros((4, 4))
    for i in range(2):
        idx = [i, i + 2]
        Q[np.ix_(idx, idx)] = g
    return F, Q


def _ca_mats(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt**2
    g = q * np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    Q = np.zeros((6, 6))
    for i in range(2):
        idx = [i, i + 2, i + 4]
        Q[np.ix_(idx, idx)] = g
    return F, Q


class KalmanFilter2D:
    """Position-measured Kalman filter (CV or CA) with variable dt."""

    def __init__(self, model: str = "ca", q: float = 1.0, r: float = 1.0):
        assert model in ("cv", "ca")
        self.model = model
        self.n = 4 if model == "cv" else 6
        self.q = q
        self.r = r
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.H = np.zeros((2, self.n))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.last_innovation: np.ndarray | None = None

    def mats(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return _cv_mats(dt, self.q) if self.model == "cv" else _ca_mats(dt, self.q)

    def init(self, z: np.ndarray, pos_var: float | None = None, vel_var: float = 100.0, acc_var: float = 100.0):
        self.x = np.zeros(self.n)
        self.x[:2] = z
        d = [pos_var if pos_var is not None else self.r] * 2 + [vel_var] * 2
        if self.n == 6:
            d += [acc_var] * 2
        self.P = np.diag(d).astype(float)

    def predict(self, dt: float) -> None:
        F, Q = self.mats(max(dt, 1e-4))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, r: float | None = None) -> None:
        R = np.eye(2) * (self.r if r is None else r)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        self.last_innovation = y

    def forecast(
        self, horizons: list[float] | np.ndarray, model: str | None = None, q: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the current estimate open-loop. Returns means (H,2) and position covariances (H,2,2).

        ``model='cv'`` forecasts a CA filter with constant velocity (acceleration dropped),
        which is usually more robust beyond ~1 s for pedestrians. ``q`` overrides the process
        noise used for the open-loop propagation (the forecast noise is fitted on validation
        data and differs from the tracking filter's noise).
        """
        model = model or self.model
        qq = self.q if q is None else q
        if model == "cv" or self.n == 4:
            x0, P0 = self.x[:4].copy(), self.P[:4, :4].copy()
            mk = lambda dt: _cv_mats(dt, qq)  # noqa: E731
        else:
            x0, P0 = self.x.copy(), self.P.copy()
            mk = lambda dt: _ca_mats(dt, qq)  # noqa: E731
        means, covs = [], []
        for h in horizons:
            F, Q = mk(float(h))
            x = F @ x0
            P = F @ P0 @ F.T + Q
            means.append(x[:2])
            covs.append(P[:2, :2])
        return np.asarray(means), np.asarray(covs)
