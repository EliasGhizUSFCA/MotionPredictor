"""Evaluation and probability calibration for trajectory forecasters.

Can be run standalone on a saved checkpoint:
    python training/evaluate.py --checkpoint models/socialmdn_zara1/model.pt --test-scene zara1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from motionpredict.forecasting import metrics as M  # noqa: E402
from motionpredict.forecasting.baseline import constant_acceleration, constant_velocity, kalman_cv  # noqa: E402
from motionpredict.forecasting.constants import DT, OBS_LEN, PRED_LEN  # noqa: E402

HORIZON_STEPS = {"1.2s": 3, "2.0s": 5, "3.2s": 8, "4.8s": 12, "5.2s": 13}


@torch.no_grad()
def predict(model, ws, batch: int = 1024) -> dict[str, np.ndarray]:
    mus, covs, logits = [], [], []
    for b in range(0, len(ws), batch):
        sl = slice(b, b + batch)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        obs = t(ws.obs[sl])
        out = model(obs, torch.ones(obs.shape[:2]), t(ws.nbrs[sl]), t(ws.nbr_mask[sl]))
        mus.append(out["mu"].numpy())
        covs.append(out["cov"].numpy())
        logits.append(out["logits"].numpy())
    return {"mu": np.concatenate(mus), "cov": np.concatenate(covs), "logits": np.concatenate(logits)}


def softmax(x, T=1.0):
    z = x / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(pred, ws) -> tuple[float, dict]:
    """Temperature scaling of mode logits against the 'closest mode' label on validation data."""
    _, _, best = M.min_ade_fde(pred["mu"], ws.fut)
    lg = pred["logits"]

    def nll(logT):
        p = softmax(lg, np.exp(logT))
        return -np.log(np.clip(p[np.arange(len(best)), best], 1e-12, 1)).mean()

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    T = float(np.exp(res.x))
    before = M.mode_calibration(softmax(lg, 1.0), best)
    after = M.mode_calibration(softmax(lg, T), best)
    return T, {
        "temperature": T,
        "val_ece_all_modes_before": before["ece_all_modes"],
        "val_ece_all_modes_after": after["ece_all_modes"],
        "val_mode_nll_before": before["mode_nll"],
        "val_mode_nll_after": after["mode_nll"],
    }


def fit_sigma_scale(pred, ws, temperature: float) -> tuple[float, dict]:
    """Scalar multiplier on predicted standard deviations, fitted by validation mixture NLL."""
    gt = ws.fut.astype(np.float64)
    mu, cov = pred["mu"].astype(np.float64), pred["cov"].astype(np.float64)
    pi = softmax(pred["logits"], temperature)
    res = minimize_scalar(lambda ls: mixture_nll(mu, cov * np.exp(2 * ls), pi, gt), bounds=(-1.5, 2.0), method="bounded")
    s = float(np.exp(res.x))
    return s, {"sigma_scale": s, "val_nll_before": mixture_nll(mu, cov, pi, gt), "val_nll_after": float(res.fun)}


def fit_kalman_q(ws, r: float = 0.01) -> tuple[float, dict]:
    """White-noise-acceleration PSD for the CV baseline, fitted by validation NLL (subsample for speed)."""
    idx = np.random.default_rng(0).choice(len(ws), size=min(3000, len(ws)), replace=False)
    obs, gt = ws.obs[idx].astype(np.float64), ws.fut[idx].astype(np.float64)
    grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    scores = {}
    for q in grid:
        m, c = kalman_cv(obs, q=q, r=r)
        scores[q] = mixture_nll(m[:, None], c[:, None], np.ones((len(gt), 1)), gt)
    best = min(scores, key=scores.get)
    return float(best), {"kalman_q": best, "val_nll_by_q": {str(k): v for k, v in scores.items()}}


def per_horizon(pred_paths, gt):
    return {h: float(np.linalg.norm(pred_paths[:, s - 1] - gt[:, s - 1], axis=-1).mean()) for h, s in HORIZON_STEPS.items()}


def mixture_nll(mu, cov, pi, gt):
    """Average per-step NLL treating steps independently given the mode (matches the model)."""
    N, K, T, _ = mu.shape
    inv = np.linalg.inv(cov)
    d = gt[:, None] - mu
    m = np.einsum("nkti,nktij,nktj->nkt", d, inv, d)
    lp = (-0.5 * m - 0.5 * np.log(np.linalg.det(cov)) - np.log(2 * np.pi)).sum(-1) + np.log(np.clip(pi, 1e-12, 1))
    mx = lp.max(-1, keepdims=True)
    return float(-(mx[:, 0] + np.log(np.exp(lp - mx).sum(-1))).mean() / T)


def evaluate_all(model, ws, temperature: float, q: float = 0.5, r: float = 0.01, sigma_scale: float = 1.0) -> dict:
    gt = ws.fut.astype(np.float64)
    t0 = time.perf_counter()
    pred = predict(model, ws)
    lat_ms = (time.perf_counter() - t0) / max(len(ws), 1) * 1000
    # single-sample latency (what the live system pays per track)
    one = type(ws)(ws.obs[:1], ws.fut[:1], ws.nbrs[:1], ws.nbr_mask[:1], ws.scene[:1])
    ts = []
    for _ in range(30):
        t1 = time.perf_counter()
        predict(model, one)
        ts.append((time.perf_counter() - t1) * 1000)

    mu, cov_raw = pred["mu"].astype(np.float64), pred["cov"].astype(np.float64)
    cov = cov_raw * sigma_scale**2
    p_raw = softmax(pred["logits"], 1.0)
    p_cal = softmax(pred["logits"], temperature)
    top = mu[np.arange(len(gt)), p_cal.argmax(-1)]
    minade, minfde, best = M.min_ade_fde(mu, gt)

    cov_by_h, cov_by_h_raw = {}, {}
    for h, s in HORIZON_STEPS.items():
        cov_by_h[h] = M.hdr_coverage(p_cal, mu[:, :, s - 1], cov[:, :, s - 1], gt[:, s - 1])
        cov_by_h_raw[h] = M.hdr_coverage(p_raw, mu[:, :, s - 1], cov_raw[:, :, s - 1], gt[:, s - 1])

    K = mu.shape[1]
    min_by_k = {}
    order = np.argsort(-p_cal, -1)
    for k in range(1, K + 1):
        sel = np.take_along_axis(mu, order[:, :k, None, None], 1)
        a, f, _ = M.min_ade_fde(sel, gt)
        min_by_k[f"top{k}"] = {"minade": float(a.mean()), "minfde": float(f.mean())}

    model_rep = {
        "n_windows": int(len(gt)),
        "ade": float(M.ade(top, gt).mean()),
        "fde": float(M.fde(top, gt).mean()),
        "ade_4.8s": float(M.ade(top[:, :12], gt[:, :12]).mean()),
        "fde_4.8s": float(M.fde(top[:, :12], gt[:, :12]).mean()),
        "minade": float(minade.mean()),
        "minfde": float(minfde.mean()),
        "minade_4.8s": float(M.min_ade_fde(mu[:, :, :12], gt[:, :12])[0].mean()),
        "minfde_4.8s": float(M.min_ade_fde(mu[:, :, :12], gt[:, :12])[1].mean()),
        "min_over_top_k": min_by_k,
        "displacement_by_horizon_top_mode": per_horizon(top, gt),
        "nll_per_step": mixture_nll(mu, cov, p_cal, gt),
        "nll_per_step_uncalibrated": mixture_nll(mu, cov_raw, p_raw, gt),
        "hdr_coverage_uncalibrated": cov_by_h_raw,
        "sigma_scale": sigma_scale,
        "mode_calibration_raw": M.mode_calibration(p_raw, best),
        "mode_calibration_calibrated": M.mode_calibration(p_cal, best),
        "hdr_coverage": cov_by_h,
        "latency_ms_batched_per_window": lat_ms,
        "latency_ms_single_window_median": float(np.median(ts)),
        "temperature": temperature,
    }

    cv = constant_velocity(ws.obs.astype(np.float64))
    ca = constant_acceleration(ws.obs.astype(np.float64))
    km, kc = kalman_cv(ws.obs.astype(np.float64), q=q, r=r)
    kal_cov = {h: M.hdr_coverage(np.ones((len(gt), 1)), km[:, None, s - 1], kc[:, None, s - 1], gt[:, s - 1]) for h, s in HORIZON_STEPS.items()}
    base = {
        "constant_velocity": {"ade": float(M.ade(cv, gt).mean()), "fde": float(M.fde(cv, gt).mean()),
                              "ade_4.8s": float(M.ade(cv[:, :12], gt[:, :12]).mean()), "fde_4.8s": float(M.fde(cv[:, :12], gt[:, :12]).mean()),
                              "displacement_by_horizon": per_horizon(cv, gt)},
        "constant_acceleration": {"ade": float(M.ade(ca, gt).mean()), "fde": float(M.fde(ca, gt).mean()),
                                  "displacement_by_horizon": per_horizon(ca, gt)},
        "kalman_cv": {"ade": float(M.ade(km, gt).mean()), "fde": float(M.fde(km, gt).mean()),
                      "ade_4.8s": float(M.ade(km[:, :12], gt[:, :12]).mean()), "fde_4.8s": float(M.fde(km[:, :12], gt[:, :12]).mean()),
                      "nll_per_step": mixture_nll(km[:, None], kc[:, None], np.ones((len(gt), 1)), gt),
                      "hdr_coverage": kal_cov, "q": q, "r": r, "displacement_by_horizon": per_horizon(km, gt)},
    }
    return {"test_scene": str(ws.scene[0]) if len(ws) else None, "dt": DT, "obs_len": OBS_LEN, "pred_len": PRED_LEN,
            "model": model_rep, "baselines": base}


def main() -> None:
    from motionpredict.forecasting.data import build_splits
    from motionpredict.forecasting.model import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-scene", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, meta = load_checkpoint(a.checkpoint)
    splits = build_splits(Path(a.data), a.test_scene)
    rep = evaluate_all(model, splits["test"], meta.get("temperature", 1.0), q=meta.get("kalman_q", 0.5), sigma_scale=meta.get("sigma_scale", 1.0))
    s = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
