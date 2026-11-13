import math

import numpy as np
import pytest

from motionpredict.calibration import camera_height_from_reference, estimate_height, fit_calibration

W, H, F = 1280, 720, 600.0


def project(X, Y, Z, hc=3.0, tilt_deg=0.0):
    """Pinhole camera at height hc looking along +Y, pitched down by tilt. Returns pixel (u, v)."""
    t = math.radians(tilt_deg)
    # world (X right, Y forward, Z up) -> camera (x right, y down, z forward)
    yc, zc = -(Z - hc), Y
    y = yc * math.cos(t) + zc * math.sin(t)
    z = -yc * math.sin(t) + zc * math.cos(t)
    return F * X / z + W / 2, F * y / z + H / 2


def norm(p):
    return [p[0] / W, p[1] / H]


GROUND = [(-2, 6), (2, 6), (2, 14), (-2, 14), (0, 9), (1, 7)]


@pytest.mark.parametrize("tilt", [0.0, -5.0])
def test_homography_maps_ground(tilt):
    img = [norm(project(x, y, 0, tilt_deg=tilt)) for x, y in GROUND]
    cal = fit_calibration(img, [list(g) for g in GROUND], camera_height_m=3.0)
    assert cal.status == "calibrated_verified" and cal.rms_error_m < 1e-4
    p = cal.to_ground(np.array([norm(project(1.5, 8.0, 0, tilt_deg=tilt))]))[0]
    assert np.allclose(p, [1.5, 8.0], atol=1e-4)
    back = cal.to_image(np.array([[1.5, 8.0]]))[0]
    assert np.allclose(back, norm(project(1.5, 8.0, 0, tilt_deg=tilt)), atol=1e-4)


def test_four_points_unverified_and_degenerate_rejected():
    img = [norm(project(x, y, 0)) for x, y in GROUND[:4]]
    cal = fit_calibration(img, [list(g) for g in GROUND[:4]])
    assert cal.status == "calibrated_unverified" and cal.camera_height_m is None
    with pytest.raises(ValueError):
        fit_calibration(img[:3], [list(g) for g in GROUND[:3]])


def test_height_horizon_ratio_zero_tilt_exact():
    img = [norm(project(x, y, 0)) for x, y in GROUND]
    cal = fit_calibration(img, [list(g) for g in GROUND], camera_height_m=3.0)
    assert abs(cal.horizon_v(0.5) - 0.5) < 1e-6  # zero tilt: horizon through the principal point
    foot = project(0.5, 6.0, 0)
    head = project(0.5, 6.0, 1.75)
    bbox = [foot[0] - 30, head[1], foot[0] + 30, foot[1]]
    kps = np.zeros((17, 3))
    kps[:, 2] = 1.0
    kps[[5, 6], :2] = [[foot[0] - 15, head[1] + 60], [foot[0] + 15, head[1] + 60]]
    kps[[11, 12], :2] = [[foot[0] - 12, head[1] + 160], [foot[0] + 12, head[1] + 160]]
    est = estimate_height(cal, bbox, kps.tolist(), W, H)
    assert est.status == "calibrated"
    assert abs(est.meters - 1.75) < 0.01
    assert est.interval_m[0] < 1.75 < est.interval_m[1]


def test_height_reference_object_and_uncalibrated():
    img = [norm(project(x, y, 0)) for x, y in GROUND]
    cal = fit_calibration(img, [list(g) for g in GROUND],
                          reference={"foot": norm(project(-1, 8, 0)), "top": norm(project(-1, 8, 2.0)), "height_m": 2.0})
    assert abs(cal.camera_height_m - 3.0) < 1e-6 and cal.camera_height_source == "reference_object"
    assert camera_height_from_reference(cal, norm(project(-1, 8, 0)), norm(project(-1, 8, 2.0)), 2.0) == pytest.approx(3.0)
    est = estimate_height(None, [0, 100, 50, 300], None, W, H)
    assert est.status == "uncalibrated" and est.meters is None


def test_height_refused_without_full_body():
    img = [norm(project(x, y, 0)) for x, y in GROUND]
    cal = fit_calibration(img, [list(g) for g in GROUND], camera_height_m=3.0)
    kps = np.zeros((17, 3)).tolist()  # nothing visible
    est = estimate_height(cal, [600, 300, 660, 500], kps, W, H)
    assert est.status == "unavailable" and est.meters is None
