"""Physics baselines.

* ``constant_velocity`` - velocity of the last observed step, extrapolated.
* ``constant_acceleration`` - finite-difference acceleration, extrapolated (often worse at long horizons).
* ``kalman_cv`` - Kalman-filtered constant-velocity model; returns a Gaussian per future step
  whose covariance is the open-loop propagated state uncertainty. This is reported as an
  *uncertainty band*, never as route probabilities.
"""

from __future__ import annotations

import numpy as np

from ..kalman import KalmanFilter2D
from .constants import DT, PRED_LEN


def constant_velocity(obs: np.ndarray, pred_len: int = PRED_LEN, n_avg: int = 1) -> np.ndarray:
    """obs (N, T, 2) -> (N, pred_len, 2)."""
    v = (obs[:, -1] - obs[:, -1 - n_avg]) / n_avg
    k = np.arange(1, pred_len + 1)[None, :, None]
    return obs[:, -1:, :] + v[:, None] * k


def constant_acceleration(obs: np.ndarray, pred_len: int = PRED_LEN) -> np.ndarray:
    v = obs[:, -1] - obs[:, -2]
    a = (obs[:, -1] - obs[:, -2]) - (obs[:, -2] - obs[:, -3])
    k = np.arange(1, pred_len + 1)[None, :, None]
    return obs[:, -1:, :] + v[:, None] * k + 0.5 * a[:, None] * k**2


def kalman_cv(
    obs: np.ndarray, q: float = 0.5, r: float = 0.01, dt: float = DT, pred_len: int = PRED_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Filter each observed track with a CV Kalman filter and forecast open loop.

    Returns means (N, pred_len, 2) and covariances (N, pred_len, 2, 2)."""
    N = obs.shape[0]
    means = np.zeros((N, pred_len, 2))
    covs = np.zeros((N, pred_len, 2, 2))
    horizons = dt * np.arange(1, pred_len + 1)
    for i in range(N):
        kf = KalmanFilter2D("cv", q=q, r=r)
        kf.init(obs[i, 0], vel_var=4.0)
        for t in range(1, obs.shape[1]):
            kf.predict(dt)
            kf.update(obs[i, t])
        means[i], covs[i] = kf.forecast(horizons)
    return means, covs
