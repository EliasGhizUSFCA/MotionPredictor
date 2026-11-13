import math

import numpy as np
import torch

from motionpredict.forecasting import metrics as M
from motionpredict.forecasting.constants import NUM_MODES, OBS_LEN, PRED_LEN
from motionpredict.forecasting.model import ModelConfig, SocialMDN, loss_fn, mixture_log_prob


def _inputs(B=4):
    torch.manual_seed(0)
    obs = torch.cumsum(torch.randn(B, OBS_LEN, 2) * 0.1 + torch.tensor([0.5, 0.1]), 1)
    return obs, torch.ones(B, OBS_LEN), torch.randn(B, 8, 4), (torch.rand(B, 8) > 0.5).float()


def test_shapes_and_constraints():
    m = SocialMDN(ModelConfig()).eval()
    obs, mask, nb, nm = _inputs()
    out = m(obs, mask, nb, nm)
    assert out["mu"].shape == (4, NUM_MODES, PRED_LEN, 2)
    assert out["cov"].shape == (4, NUM_MODES, PRED_LEN, 2, 2)
    p = torch.softmax(out["logits"], -1)
    assert torch.allclose(p.sum(-1), torch.ones(4))
    s = out["sigma_ego"]
    assert (s[:, :, 1:] >= s[:, :, :-1] - 1e-6).all(), "uncertainty must not shrink with horizon"
    loss, parts = loss_fn(out, torch.zeros(4, PRED_LEN, 2) + obs[:, -1:])
    assert torch.isfinite(loss)


def test_rotation_equivariance():
    """Heading-frame canonicalisation => rotating the scene rotates the forecast."""
    m = SocialMDN(ModelConfig(social=True)).eval()
    obs, mask, nb, nm = _inputs(2)
    th = 0.7
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    nb_r = nb.clone()
    nb_r[..., :2] = nb[..., :2] @ R.T
    nb_r[..., 2:] = nb[..., 2:] @ R.T
    with torch.no_grad():
        a = m(obs, mask, nb, nm)
        b = m(obs @ R.T, mask, nb_r, nm)
    assert torch.allclose(a["mu"] @ R.T, b["mu"], atol=1e-4)
    assert torch.allclose(a["logits"], b["logits"], atol=1e-4)


def test_short_history_mask():
    m = SocialMDN().eval()
    obs, mask, _, _ = _inputs(1)
    mask[:, :5] = 0
    obs[:, :5] = obs[:, 5:6]
    with torch.no_grad():
        out = m(obs, mask)
    assert torch.isfinite(out["mu"]).all()


def test_mixture_log_prob_matches_gaussian():
    m = SocialMDN(ModelConfig(num_modes=1)).eval()
    obs, mask, _, _ = _inputs(1)
    with torch.no_grad():
        out = m(obs, mask)
        lp = mixture_log_prob(out, out["mu"][:, 0])
    s = out["sigma_ego"][0, 0]
    expected = (-torch.log(s) - 0.5 * math.log(2 * math.pi)).sum()
    assert torch.allclose(lp[0], expected, atol=1e-4)


def test_metrics_calibration_and_coverage():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), size=4000)
    best = np.array([rng.choice(3, p=pi) for pi in p])  # labels drawn from the predicted probs
    cal = M.mode_calibration(p, best)
    assert cal["ece_all_modes"] < 0.03
    N = 3000
    mu = np.zeros((N, 1, 2))
    cov = np.tile(np.eye(2) * 0.25, (N, 1, 1, 1))
    gt = rng.normal(0, 0.5, (N, 2))
    cv = M.hdr_coverage(np.ones((N, 1)), mu, cov, gt)
    assert abs(cv["0.50"] - 0.5) < 0.04 and abs(cv["0.90"] - 0.9) < 0.03
    a, f, b = M.min_ade_fde(np.zeros((1, 2, 3, 2)) + np.array([0, 1.0])[None, :, None, None], np.ones((1, 3, 2)))
    assert b[0] == 1 and a[0] == 0
