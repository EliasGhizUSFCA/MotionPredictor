"""Ground-plane calibration and approximate monocular height estimation.

Ground plane
    A homography H maps normalised image coordinates (u, v in [0,1]) to metric ground-plane
    coordinates. It is fitted from >= 4 user-marked ground points with known positions
    (e.g. corners of a floor tile pattern or a measured rectangle). With exactly 4 points the
    fit is exact and therefore *unverified*; with >= 5 points the residual is reported and the
    calibration is marked ``verified`` when the RMS residual is below 0.15 m.

Horizon
    The image of the ground plane's line at infinity is the third row of H
    (l = H^T [0,0,1]^T), i.e. h31*u + h32*v + h33 = 0.

Height (single-view metrology, Hoiem et al. 2006 "horizon ratio")
    For a camera with small roll / tilt, an upright object whose foot is at image row v_f and
    top at v_t has height  Z = Hc * (v_f - v_t) / (v_f - v_h),  where v_h is the horizon row
    at the foot's column and Hc is the camera height above the ground. Hc is either entered by
    the user or recovered from a reference object of known height. This is an approximation:
    it ignores lens distortion and assumes image verticals are near-parallel. We therefore
    report an interval and only when the whole body is visible and upright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .schemas import HeightEstimate

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
L_ANK, R_ANK = 15, 16


@dataclass
class GroundCalibration:
    H: np.ndarray  # normalised image -> ground (meters)
    image_points: list[list[float]]
    world_points: list[list[float]]
    rms_error_m: float | None
    camera_height_m: float | None = None
    camera_height_source: str | None = None
    height_tolerance: float = 0.05
    warnings: list[str] = field(default_factory=list)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    @property
    def status(self) -> str:
        if self.rms_error_m is None:
            return "calibrated_unverified"
        return "calibrated_verified" if self.rms_error_m < 0.15 else "calibrated_poor_fit"

    def horizon_v(self, u: float) -> float | None:
        h = self.H[2]
        if abs(h[1]) < 1e-9:
            return None
        return float(-(h[0] * u + h[2]) / h[1])

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        p = np.c_[uv, np.ones(len(uv))] @ self.H.T
        return p[:, :2] / p[:, 2:3]

    @property
    def _w_sign(self) -> float:
        """Sign of the homogeneous coordinate for visible ground points (H is defined up to scale)."""
        ref = self.to_ground(np.array([[0.5, 1.0]]))  # bottom-centre of the image is on visible ground
        w = (np.c_[ref, [[1.0]]] @ self.H_inv.T)[0, 2]
        return 1.0 if w >= 0 else -1.0

    def to_image(self, xy: np.ndarray) -> np.ndarray:
        """Ground -> normalised image. Points beyond the horizon (not visible) become NaN."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        p = np.c_[xy, np.ones(len(xy))] @ self.H_inv.T
        w = p[:, 2:3]
        out = p[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)
        out[(w[:, 0] * self._w_sign <= 0)] = np.nan
        return out

    def to_dict(self) -> dict:
        hz = [self.horizon_v(0.0), self.horizon_v(1.0)]
        return {
            "status": self.status,
            "coordinate_system": "ground_plane_meters",
            "homography": self.H.tolist(),
            "homography_inv": self.H_inv.tolist(),
            "image_points": self.image_points,
            "world_points": self.world_points,
            "rms_error_m": self.rms_error_m,
            "horizon_v": hz,
            "camera_height_m": self.camera_height_m,
            "camera_height_source": self.camera_height_source,
            "height_estimation": self.camera_height_m is not None and hz[0] is not None,
            "warnings": self.warnings,
        }


def fit_calibration(
    image_points: list[list[float]],
    world_points: list[list[float]],
    camera_height_m: float | None = None,
    reference: dict | None = None,
    height_tolerance: float = 0.05,
) -> GroundCalibration:
    img = np.asarray(image_points, float)
    wld = np.asarray(world_points, float)
    if len(img) < 4 or len(img) != len(wld):
        raise ValueError("need >= 4 matching image/world ground points")
    if np.any(img < -0.05) or np.any(img > 1.05):
        raise ValueError("image points must be normalised to [0,1]")
    H, _ = cv2.findHomography(img, wld, 0)
    if H is None or not np.isfinite(H).all():
        raise ValueError("degenerate point configuration (are 3 points collinear?)")
    H = H / H[2, 2]
    warnings: list[str] = []
    rms = None
    if len(img) >= 5:
        p = np.c_[img, np.ones(len(img))] @ H.T
        rms = float(np.sqrt(np.mean(np.sum((p[:, :2] / p[:, 2:3] - wld) ** 2, -1))))
    else:
        warnings.append("Exactly 4 points: homography is exactly determined and cannot be verified. Add a 5th point to measure error.")
    cal = GroundCalibration(H, img.tolist(), wld.tolist(), rms, height_tolerance=height_tolerance, warnings=warnings)
    # every ground point must be in front of the horizon
    test = cal.to_ground(img)
    if not np.isfinite(test).all():
        raise ValueError("calibration places marked points beyond the horizon")
    if camera_height_m:
        cal.camera_height_m = float(camera_height_m)
        cal.camera_height_source = "user_entered"
    elif reference:
        hc = camera_height_from_reference(cal, reference["foot"], reference["top"], float(reference["height_m"]))
        if hc is None:
            warnings.append("Reference object could not be used (top at or above horizon).")
        else:
            cal.camera_height_m = hc
            cal.camera_height_source = "reference_object"
    if cal.camera_height_m is None:
        warnings.append("No camera height / reference object: metric height estimation disabled.")
    elif cal.horizon_v(0.5) is None:
        warnings.append("Horizon is at infinity (near top-down view): height estimation disabled.")
    return cal


def camera_height_from_reference(cal: GroundCalibration, foot, top, height_m: float) -> float | None:
    vh = cal.horizon_v(foot[0])
    if vh is None:
        return None
    num, den = foot[1] - vh, foot[1] - top[1]
    if den <= 1e-6 or num <= 0:
        return None
    return float(height_m * num / den)


def keypoints_full_body(kps: list | None, thr: float = 0.5) -> tuple[bool, str]:
    if not kps:
        return False, "no pose keypoints"
    k = np.asarray(kps)
    head = max(k[NOSE, 2], k[L_EYE, 2], k[R_EYE, 2]) > thr
    feet = max(k[L_ANK, 2], k[R_ANK, 2]) > thr
    if not head:
        return False, "head not visible"
    if not feet:
        return False, "feet not visible"
    sh = (k[L_SHO, :2] + k[R_SHO, :2]) / 2
    hp = (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    torso = hp - sh
    if np.linalg.norm(torso) < 1e-3 or abs(torso[1]) < 0.7 * np.linalg.norm(torso):
        return False, "person not upright"
    return True, ""


def estimate_height(
    cal: GroundCalibration | None, bbox: list[float], kps: list | None, frame_w: int, frame_h: int
) -> HeightEstimate:
    x1, y1, x2, y2 = bbox
    rel = (y2 - y1) / frame_h
    if cal is None:
        return HeightEstimate(status="uncalibrated", relative_image_height=rel, reason="camera not calibrated - metric height unknown")
    if cal.camera_height_m is None:
        return HeightEstimate(status="unavailable", relative_image_height=rel, reason="calibration has no camera height / reference object")
    if y1 <= 2 or y2 >= frame_h - 2:
        return HeightEstimate(status="unavailable", relative_image_height=rel, reason="person cut off by frame edge")
    ok, why = keypoints_full_body(kps)
    if not ok:
        return HeightEstimate(status="unavailable", relative_image_height=rel, reason=why)
    uf, vf, vt = ((x1 + x2) / 2) / frame_w, y2 / frame_h, y1 / frame_h
    vh = cal.horizon_v(uf)
    if vh is None or vf - vh <= 0.02:
        return HeightEstimate(status="unavailable", relative_image_height=rel, reason="foot too close to horizon")

    def h_of(vf_, vt_, vh_, hc_):
        return hc_ * (vf_ - vt_) / (vf_ - vh_)

    z = h_of(vf, vt, vh, cal.camera_height_m)
    # first-order uncertainty: bbox edges +-3% of box height, horizon +-0.5% of frame,
    # camera height +- tolerance, plus 4% model error (tilt / bbox-vs-head-top bias)
    sb = 0.03 * (vf - vt)
    eps = 1e-5
    grads = [
        ((h_of(vf + eps, vt, vh, cal.camera_height_m) - z) / eps, sb),
        ((h_of(vf, vt + eps, vh, cal.camera_height_m) - z) / eps, sb),
        ((h_of(vf, vt, vh + eps, cal.camera_height_m) - z) / eps, 0.005),
        (z / cal.camera_height_m, cal.height_tolerance * cal.camera_height_m),
    ]
    var = sum((g * s) ** 2 for g, s in grads) + (0.04 * z) ** 2
    sd = math.sqrt(var)
    if not (0.3 < z < 2.6):
        return HeightEstimate(status="unavailable", relative_image_height=rel, reason=f"implausible estimate ({z:.2f} m) - check calibration")
    return HeightEstimate(
        status="calibrated",
        meters=round(z, 3),
        interval_m=[round(z - 1.645 * sd, 3), round(z + 1.645 * sd, 3)],
        relative_image_height=rel,
        reason="horizon-ratio single-view estimate (approximate)",
    )
