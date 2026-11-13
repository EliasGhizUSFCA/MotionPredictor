"""Scoring of one stored forecast against the observed future.

All numbers are computed in the forecast's own coordinate system (pixels or ground-plane
meters). In pixel mode errors are additionally reported in *body heights* (error / the
person's median bounding-box height at prediction time) so that people at different
distances from the camera are comparable; this is a normalisation, not a metric unit.

Definitions (per forecast, per horizon h, evaluation times on the forecast's 0.2 s grid in (0, h]):
* ADE_top1 / FDE_top1 - error of the hypothesis with the highest probability/weight
  (what a user would act on). For the physics baseline this is the CV path.
* minADE@K / minFDE@K  - error of the best of K hypotheses chosen *with hindsight*
  (oracle). Reported separately; it does not mean the model picked that path in advance.
* wADE                 - probability-weighted ADE  sum_k p_k ADE_k (only with probabilities).
* miss_top1 / miss@K   - FDE_top1 > threshold / minFDE@K > threshold.
* NLL_final            - negative log density of the true final position under the
  predicted (mixture of) Gaussian(s) at h. Lower is better; comparable across models only
  within the same coordinate system.
* in_hdr50 / in_hdr90  - whether the true final position lies inside the 50% / 90% highest
  density region of the predicted distribution (coverage should match the nominal level).
"""

from __future__ import annotations

import math

import numpy as np

from ..forecasting import metrics as M

MISS_THRESHOLD = {"ground_plane_meters": 2.0, "image_pixels_body_heights": 1.0}


def cov_from_sigma(sig: list[float]) -> np.ndarray:
    a, b, ang = sig
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return R @ np.diag([max(a, 1e-4) ** 2, max(b, 1e-4) ** 2]) @ R.T


def interp_path(times: np.ndarray, pts: np.ndarray, at: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(at, times, pts[:, 0]), np.interp(at, times, pts[:, 1])], -1)


def score_forecast(
    model_out: dict, gt_times: np.ndarray, gt_pts: np.ndarray, horizon: float, coord: str, body_px: float, seed: int = 0
) -> dict:
    trajs = model_out["trajectories"]
    if model_out["prediction_source"] == "physics_baseline":
        trajs = [t for t in trajs if t["id"] == "CV"]  # CA has no fitted distribution; excluded from scoring
    times = np.asarray(trajs[0]["times"])
    eval_t = times[(times > 1e-6) & (times <= horizon + 1e-6)]
    gt = interp_path(gt_times, gt_pts, eval_t)
    errs, finals = [], []
    for tr in trajs:
        p = interp_path(np.asarray(tr["times"]), np.asarray(tr["points"]), eval_t)
        e = np.linalg.norm(p - gt, axis=-1)
        errs.append(e.mean())
        finals.append(e[-1])
    errs, finals = np.asarray(errs), np.asarray(finals)
    probs = [t["probability"] for t in trajs]
    has_p = all(p is not None for p in probs)
    top = int(np.argmax(probs)) if has_p else 0
    best = int(np.argmin(errs))
    norm = body_px if coord == "image_pixels" else 1.0
    thr = MISS_THRESHOLD["ground_plane_meters"] if coord == "ground_plane_meters" else MISS_THRESHOLD["image_pixels_body_heights"] * body_px

    # distribution at the final evaluation time
    idx = [int(np.argmin(np.abs(np.asarray(t["times"]) - eval_t[-1]))) for t in trajs]
    mu = np.asarray([interp_path(np.asarray(t["times"]), np.asarray(t["points"]), eval_t[-1:])[0] for t in trajs])
    cov = np.asarray([cov_from_sigma(t["sigma"][i]) for t, i in zip(trajs, idx)])
    pi = np.asarray(probs, float) if has_p else np.ones(1)
    if has_p:
        pi = pi / pi.sum()  # renormalise over the stored (shown) modes
    gtf = gt[-1]
    lp = M.gaussian_mixture_logpdf(gtf[None, None], pi[None], mu[None], cov[None])[0, 0]
    hdr = M.hdr_coverage(pi[None], mu[None], cov[None], gtf[None], alphas=(0.5, 0.9), n_samples=400, seed=seed)

    out = {
        "n_eval_points": int(len(eval_t)),
        "ade_top1": float(errs[top]),
        "fde_top1": float(finals[top]),
        "min_ade": float(errs.min()),
        "min_fde": float(finals.min()),
        "k": len(trajs),
        "w_ade": float(np.dot(pi, errs)) if has_p else None,
        "miss_top1": bool(finals[top] > thr),
        "miss_k": bool(finals.min() > thr),
        "nll_final": float(-lp),
        "in_hdr50": hdr["0.50"] > 0.5,
        "in_hdr90": hdr["0.90"] > 0.5,
        "top_id": trajs[top]["id"],
        "top_prob": probs[top],
        "best_id": trajs[best]["id"],
        "mode_probs": {t["id"]: t["probability"] for t in trajs} if has_p else None,
        "miss_threshold": thr,
    }
    if coord == "image_pixels":
        for k in ("ade_top1", "fde_top1", "min_ade", "min_fde"):
            out[k + "_bh"] = out[k] / norm
    return out
