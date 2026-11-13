"""Live forecasting service used by the pipeline.

Two sources, always labelled in the output:

* ``physics_baseline`` - open-loop propagation of the track's Kalman filter (constant velocity,
  plus a constant-acceleration variant). Returned with ``probability=None`` and Gaussian
  uncertainty bands; no route probabilities are fabricated.
* ``trained_model`` - SocialMDN checkpoint. Mode probabilities are reported as
  ``calibrated_probability`` only when ALL of these hold: (a) the checkpoint carries a temperature
  fitted on validation data, its held-out ECE over all modes is below ``ECE_VALIDATED`` AND its
  held-out top-mode ECE is below ``ECE_TOP_VALIDATED`` (the top path's % is the number the UI leads
  with), (b) the live coordinates are metric ground-plane coordinates (the training domain), and
  (c) the camera calibration is ``calibrated_verified`` or the exact virtual calibration of the
  dataset replay. Otherwise they are labelled ``uncalibrated_weight``. The reason is always given
  in ``Forecast.calibration_status``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from ..schemas import Forecast, Trajectory
from .constants import DT, MAX_NEIGHBORS, OBS_LEN, PRED_LEN
from .model import load_checkpoint

ECE_VALIDATED = 0.05  # held-out ECE over all modes (pooled)
ECE_TOP_VALIDATED = 0.10  # held-out ECE of the top mode only
VIRTUAL_REPLAY_CALIBRATION = "virtual_dataset_replay"  # exact calibration of the dataset replay's top-down view
MODELS_DIR = Path(os.environ.get("MP_MODELS_DIR", Path(__file__).resolve().parents[3] / "models"))
DEFAULT_CKPT = os.environ.get("MP_FORECAST_CKPT", str(MODELS_DIR / "socialmdn_zara1" / "model.pt"))
MODE_LABELS = "ABCDEFGH"
UI_HORIZONS = [0.5, 1.0, 2.0, 3.0, 5.0]


def ellipse_params(cov: np.ndarray) -> list[float]:
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    major = v[:, 1]
    return [float(math.sqrt(w[1])), float(math.sqrt(w[0])), float(math.atan2(major[1], major[0]))]


def dense_times(horizon: float, step: float = 0.2) -> np.ndarray:
    """0, step, 2*step, ... with the exact horizon as the last time (e.g. 0.5 -> 0, 0.2, 0.4, 0.5)."""
    n = max(int(math.ceil(horizon / step - 1e-6)), 1)
    t = np.minimum(step * np.arange(0, n + 1), horizon)
    return np.round(t, 3)


class Forecaster:
    def __init__(self, checkpoint: str | None = DEFAULT_CKPT, device: str = "cpu"):
        self.device = device
        self.model = None
        self.meta: dict = {}
        self.load_error: str | None = None
        self.report: dict | None = None
        ck = checkpoint
        if ck and Path(ck).exists():
            try:
                self.model, self.meta = load_checkpoint(ck, device)
                rp = Path(ck).parent / "report.json"
                if rp.exists():
                    self.report = json.loads(rp.read_text())
            except Exception as e:  # pragma: no cover - surfaced in /api/health
                self.load_error = f"{type(e).__name__}: {e}"
        else:
            self.load_error = f"checkpoint not found: {ck} (run training/train.py)"

    # ------------------------------------------------------------------ info
    @property
    def available(self) -> bool:
        return self.model is not None

    def _checkpoint_gate(self) -> tuple[bool, list[str]]:
        """Checkpoint-level probability validation: (passed, reasons it failed)."""
        if not self.available:
            return False, ["no trained model loaded"]
        cal = self.meta.get("calibration", {}) or {}
        why = []
        if "temperature" not in self.meta:
            why.append("checkpoint has no validation-fitted temperature")
        ece_all, ece_top = cal.get("test_ece_all_modes"), cal.get("test_ece_top_mode")
        if ece_all is None:
            why.append("checkpoint reports no held-out all-modes ECE")
        elif ece_all >= ECE_VALIDATED:
            why.append(f"held-out all-modes ECE {ece_all:.3f} >= {ECE_VALIDATED}")
        if ece_top is None:
            why.append("checkpoint reports no held-out top-mode ECE")
        elif ece_top >= ECE_TOP_VALIDATED:
            why.append(f"held-out top-mode ECE {ece_top:.3f} >= {ECE_TOP_VALIDATED}")
        return not why, why

    @property
    def validated(self) -> bool:
        """Checkpoint passed both held-out ECE gates (all modes and top mode)."""
        return self._checkpoint_gate()[0]

    def probability_label(self, coord: str, camera_calibration: str | None) -> tuple[str, str]:
        """(probability_type, calibration_status reason) for live output - see the module docstring.

        ``camera_calibration`` is the camera calibration status (``calibrated_verified`` ...), or
        ``VIRTUAL_REPLAY_CALIBRATION`` for the dataset replay's exact virtual camera."""
        ok, why = self._checkpoint_gate()
        if coord != "ground_plane_meters":
            why.append("live coordinates are image pixels, not the metric ground plane the model was trained on")
        elif camera_calibration not in ("calibrated_verified", VIRTUAL_REPLAY_CALIBRATION):
            why.append(f"camera calibration is '{camera_calibration or 'uncalibrated'}', not 'calibrated_verified'")
        if why:
            return "uncalibrated_weight", "uncalibrated: " + "; ".join(why)
        cal = self.meta.get("calibration", {})
        scene = self.meta.get("held_out_test_scene", "held-out")
        where = ("dataset replay with exact virtual calibration" if camera_calibration == VIRTUAL_REPLAY_CALIBRATION
                 else "verified camera calibration; this camera/domain itself has not been validated")
        return "calibrated_probability", (f"validated_on_heldout_data: {scene} held-out ECE all modes "
                                          f"{cal['test_ece_all_modes']:.3f}, top mode {cal['test_ece_top_mode']:.3f}; {where}")

    def info(self) -> dict:
        ok, why = self._checkpoint_gate()
        cal = self.meta.get("calibration", {}) or {}
        return {
            "available": self.available,
            "load_error": self.load_error,
            "version": self.meta.get("version"),
            "meta": self.meta,
            "probability_calibration_validated": ok,
            "probability_gate": {"passed": ok, "reasons": why,
                                 "test_ece_all_modes": cal.get("test_ece_all_modes"), "test_ece_top_mode": cal.get("test_ece_top_mode"),
                                 "requires": "metric coordinates and a calibrated_verified camera (or dataset replay)"},
            "ece_threshold": ECE_VALIDATED,
            "ece_top_mode_threshold": ECE_TOP_VALIDATED,
        }

    # -------------------------------------------------------------- baseline
    @property
    def kalman_q(self) -> float:
        """Forecast process noise (m^2/s^3) fitted on validation data; 0.1 if no fitted value."""
        return float(self.meta.get("kalman_q", 0.1))

    def baseline(self, track_id: int, kf, horizon: float, coord: str, warnings: list[str], hist_steps: int,
                 unit_scale: float = 1.0, num_hypotheses: int = 2) -> Forecast:
        """``unit_scale`` = working units per meter (1 in meters, px-per-pseudo-meter in image mode).

        The baseline has at most 2 motion assumptions: CV only when ``num_hypotheses`` is 1, CV + CA otherwise."""
        times = dense_times(horizon)[1:]
        trajs = []
        q_cv = self.kalman_q * unit_scale**2
        variants = [("CV", "cv", q_cv), ("CA", "ca", None)][: 1 if num_hypotheses <= 1 else 2]
        for rank, (name, model, q) in enumerate(variants):
            means, covs = kf.forecast(times, model=model, q=q)
            pts = np.vstack([kf.x[:2], means])
            if name == "CV":
                sig = [ellipse_params(kf.P[:2, :2])] + [ellipse_params(c) for c in covs]
            else:  # CA variant: mean path only - its process noise is not fitted, so no band is claimed
                sig = [[0.0, 0.0, 0.0]] * len(pts)
            trajs.append(
                Trajectory(id=name, rank=rank, probability=None, times=[0.0] + times.tolist(),
                           points=pts.tolist(), sigma=sig)
            )
        return Forecast(
            track_id=track_id, horizon_seconds=horizon, coordinate_system=coord, prediction_source="physics_baseline",
            model_version="kalman-cv/ca-v1", probability_type="none", calibration_status="not_applicable",
            trajectories=trajs, history_steps_used=hist_steps,
            warnings=warnings + ["Physics baseline: the band shows the constant-velocity Kalman forecast uncertainty "
                                 "(noise fitted on validation data). CA is an alternative motion assumption drawn without a band. "
                                 "Neither carries a route probability."]
            + ([f"{num_hypotheses} hypotheses requested, but the physics baseline has at most 2 motion assumptions (CV and CA)."]
               if num_hypotheses > 2 else
               ["1 hypothesis requested: showing the constant-velocity path only (the baseline has at most 2 motion assumptions)."]
               if num_hypotheses <= 1 else []),
        )

    # --------------------------------------------------------------- learned
    @torch.no_grad()
    def learned_batch(self, items: list[dict], horizon: float, coord: str, k_show: int,
                      camera_calibration: str | None = None) -> list[Forecast]:
        """items: [{track_id, obs (8,2), n_valid, nbrs (N,4), scale, warnings}] in working units.
        Neighbours should be sorted nearest-first (as in training); only the first MAX_NEIGHBORS are used.
        ``camera_calibration``: see ``probability_label``.

        ``scale`` converts working units to model units (meters): 1.0 when calibrated,
        1.7 / bbox_height_px in image mode (body-height normalisation; see README)."""
        if not items:
            return []
        B = len(items)
        obs = np.zeros((B, OBS_LEN, 2), np.float32)
        mask = np.zeros((B, OBS_LEN), np.float32)
        nb = np.zeros((B, MAX_NEIGHBORS, 4), np.float32)
        nm = np.zeros((B, MAX_NEIGHBORS), np.float32)
        for i, it in enumerate(items):
            s = it["scale"]
            obs[i] = it["obs"] * s
            mask[i, OBS_LEN - it["n_valid"] :] = 1
            n = min(len(it["nbrs"]), MAX_NEIGHBORS)
            if n:
                nb[i, :n] = np.asarray(it["nbrs"][:n]) * s
                nm[i, :n] = 1
        t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        out = self.model(t(obs), t(mask), t(nb), t(nm))
        temp = float(self.meta.get("temperature", 1.0))
        probs = torch.softmax(out["logits"] / temp, -1).cpu().numpy()
        mu = out["mu"].cpu().numpy()
        cov = out["cov"].cpu().numpy() * float(self.meta.get("sigma_scale", 1.0)) ** 2
        model_t = DT * np.arange(1, PRED_LEN + 1)
        times = dense_times(horizon)
        metric = coord == "ground_plane_meters"
        prob_type, cal_status = self.probability_label(coord, camera_calibration)
        res = []
        for i, it in enumerate(items):
            s = it["scale"]
            origin = obs[i, -1] / s
            order = np.argsort(-probs[i])
            trajs = []
            for rank, k in enumerate(order[:k_show]):
                m = np.vstack([obs[i, -1], mu[i, k]]) / s  # (T+1,2) working units
                c = np.concatenate([np.eye(2)[None] * 1e-6, cov[i, k]]) / (s * s)
                tt = np.concatenate([[0.0], model_t])
                px = np.interp(times, tt, m[:, 0])
                py = np.interp(times, tt, m[:, 1])
                ci = np.stack([np.stack([np.interp(times, tt, c[:, a, b]) for b in range(2)], -1) for a in range(2)], -2)
                pts = np.stack([px, py], -1)
                pts[0] = origin
                trajs.append(
                    Trajectory(id=MODE_LABELS[k], rank=rank, probability=float(probs[i, k]), times=times.tolist(),
                               points=pts.tolist(), sigma=[ellipse_params(cc) for cc in ci])
                )
            w = list(it.get("warnings", []))
            if not metric:
                w.append("Learned model trained on metric ground-plane data; applied here to image-plane coordinates "
                         "(normalised by body height). Perspective makes this out-of-distribution - weights are uncalibrated.")
            if it["n_valid"] < OBS_LEN:
                w.append(f"Short history ({it['n_valid']}/{OBS_LEN} steps, {DT * (it['n_valid'] - 1):.1f}s) - forecast less reliable.")
            res.append(Forecast(
                track_id=it["track_id"], horizon_seconds=horizon, coordinate_system=coord, prediction_source="trained_model",
                model_version=self.meta.get("version", "unknown"), probability_type=prob_type, calibration_status=cal_status,
                trajectories=trajs, residual_probability=float(probs[i, order[k_show:]].sum()), history_steps_used=it["n_valid"],
                warnings=w,
            ))
        return res
