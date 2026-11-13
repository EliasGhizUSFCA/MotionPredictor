import numpy as np

from motionpredict.forecasting.baseline import constant_acceleration, constant_velocity, kalman_cv
from motionpredict.kalman import KalmanFilter2D


def test_ca_filter_recovers_velocity_with_irregular_dt():
    rng = np.random.default_rng(0)
    kf = KalmanFilter2D("ca", q=0.5, r=0.01**2)
    t, v = 0.0, np.array([1.2, -0.4])
    kf.init(np.zeros(2))
    for _ in range(60):
        dt = rng.uniform(0.02, 0.12)
        t += dt
        kf.predict(dt)
        kf.update(v * t + rng.normal(0, 0.01, 2))
    assert np.allclose(kf.x[2:4], v, atol=0.1)


def test_forecast_uncertainty_grows_with_horizon():
    kf = KalmanFilter2D("cv", q=0.1, r=0.01)
    kf.init(np.zeros(2))
    for i in range(1, 10):
        kf.predict(0.4)
        kf.update(np.array([0.5 * i, 0.0]))
    m, c = kf.forecast([0.5, 1, 2, 3, 5])
    tr = [np.trace(x) for x in c]
    assert all(b > a for a, b in zip(tr, tr[1:]))
    assert np.allclose(m[-1], [4.5 + 0.5 / 0.4 * 5, 0], atol=0.3)


def test_baselines_exact_on_straight_line():
    obs = np.stack([np.arange(8) * 0.5, np.zeros(8)], -1)[None].astype(float)
    cv = constant_velocity(obs, pred_len=5)
    assert np.allclose(cv[0, :, 0], 3.5 + 0.5 * np.arange(1, 6))
    ca = constant_acceleration(obs, pred_len=5)
    assert np.allclose(ca, cv)
    km, kc = kalman_cv(obs, q=0.01, r=1e-4, pred_len=5)
    assert np.allclose(km, cv, atol=0.05)
    assert kc.shape == (1, 5, 2, 2)
