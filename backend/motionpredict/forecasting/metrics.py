"""Forecast evaluation metrics (all distances in the dataset's units, meters for ETH/UCY).

* ADE / FDE          - average / final L2 displacement of a single (most likely) prediction.
* minADE_K / minFDE_K - best of K hypotheses (oracle selection; measures coverage of modes).
* NLL                - negative log-likelihood of the ground-truth future under the predicted
                       Gaussian mixture, reported per future step (lower is better).
* Mode ECE / Brier   - calibration of mode probabilities against the event "mode k is the
                       hypothesis closest to the ground truth" (lowest ADE). This is what the UI
                       percentages mean: "probability that this path is the best match".
* HDR coverage       - fraction of ground-truth positions at a horizon that fall inside the
                       predicted mixture's alpha highest-density region (Monte-Carlo estimate).
                       Calibrated spatial uncertainty gives coverage ~= alpha.
"""

from __future__ import annotations

import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(-1)


def fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred[..., -1, :] - gt[..., -1, :], axis=-1)


def min_ade_fde(modes: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """modes (N,K,T,2), gt (N,T,2). Returns minADE, minFDE, best-mode index (by ADE)."""
    e = np.linalg.norm(modes - gt[:, None], axis=-1)  # N,K,T
    a = e.mean(-1)
    best = a.argmin(-1)
    return a.min(-1), e[..., -1].min(-1), best


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0, 1, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        table.append({"bin": [float(lo), float(hi)], "count": int(m.sum()), "mean_confidence": float(c), "empirical_frequency": float(acc)})
    return float(ece), table


def mode_calibration(probs: np.ndarray, best: np.ndarray) -> dict:
    """probs (N,K) mode probabilities, best (N,) index of closest mode."""
    N, K = probs.shape
    onehot = np.eye(K)[best]
    # all-mode reliability: every (sample, mode) pair is a prediction "this mode is best"
    ece_all, table = expected_calibration_error(probs.ravel(), onehot.ravel())
    top = probs.argmax(-1)
    ece_top, _ = expected_calibration_error(probs.max(-1), (top == best).astype(float))
    brier = float(((probs - onehot) ** 2).sum(-1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(N), best], 1e-9, 1)).mean())
    return {
        "ece_all_modes": ece_all,
        "ece_top_mode": ece_top,
        "brier": brier,
        "mode_nll": nll,
        "top_mode_accuracy": float((top == best).mean()),
        "reliability_table": table,
    }


def gaussian_mixture_logpdf(x: np.ndarray, pi: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """x (N,M,2), pi (N,K), mu (N,K,2), cov (N,K,2,2) -> (N,M) log densities."""
    inv = np.linalg.inv(cov)  # N,K,2,2
    det = np.linalg.det(cov)
    d = x[:, None] - mu[:, :, None]  # N,K,M,2
    m = np.einsum("nkmi,nkij,nkmj->nkm", d, inv, d)
    lp = -0.5 * m - 0.5 * np.log(det)[..., None] - np.log(2 * np.pi)
    lp = lp + np.log(np.clip(pi, 1e-12, 1))[..., None]
    mx = lp.max(1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))[:, 0]


def hdr_coverage(
    pi: np.ndarray, mu: np.ndarray, cov: np.ndarray, gt: np.ndarray, alphas=(0.5, 0.9), n_samples: int = 400, seed: int = 0
) -> dict[str, float]:
    """Coverage of alpha-HDRs at a single horizon. mu (N,K,2), cov (N,K,2,2), gt (N,2)."""
    rng = np.random.default_rng(seed)
    N, K = pi.shape
    comp = np.array([rng.choice(K, size=n_samples, p=p / p.sum()) for p in pi])  # N,S
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
    eps = rng.standard_normal((N, n_samples, 2))
    Ls = L[np.arange(N)[:, None], comp]  # N,S,2,2
    samples = mu[np.arange(N)[:, None], comp] + np.einsum("nsij,nsj->nsi", Ls, eps)
    ls = gaussian_mixture_logpdf(samples, pi, mu, cov)  # N,S
    lg = gaussian_mixture_logpdf(gt[:, None], pi, mu, cov)[:, 0]
    out = {}
    for a in alphas:
        thr = np.quantile(ls, 1 - a, axis=1)
        out[f"{a:.2f}"] = float((lg >= thr).mean())
    return out
