"""Train, calibrate and evaluate the SocialMDN forecaster on one ETH/UCY leave-one-scene-out fold.

Example:
    python training/train.py --config configs/model.yaml --test-scene zara1 --out models/socialmdn_zara1

Outputs in --out:
    model.pt          checkpoint (config + weights + metadata incl. calibration temperature)
    report.json       held-out test metrics for the model and the physics baselines
    train_log.json    per-epoch training / validation losses
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from motionpredict.forecasting.constants import DT, OBS_LEN, PRED_LEN  # noqa: E402
from motionpredict.forecasting.data import augment, build_splits, left_pad_mask  # noqa: E402
from motionpredict.forecasting.model import ModelConfig, SocialMDN, loss_fn  # noqa: E402
from training.calibrate import calibrate_and_report  # noqa: E402


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_batch(ws, idx, rng, train: bool, short_hist_prob: float):
    obs, fut, nb = ws.obs[idx].copy(), ws.fut[idx].copy(), ws.nbrs[idx].copy()
    nm = ws.nbr_mask[idx].copy()
    mask = np.ones((len(idx), OBS_LEN), np.float32)
    if train:
        for i in range(len(idx)):
            obs[i], fut[i], nb[i] = augment(obs[i], fut[i], nb[i], rng)
            if rng.random() < short_hist_prob:
                obs[i], mask[i] = left_pad_mask(obs[i], int(rng.integers(2, OBS_LEN)))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    return t(obs), t(mask), t(nb), t(nm), t(fut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/model.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data/raw"))
    ap.add_argument("--test-scene", default="zara1", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    out = Path(args.out or ROOT / f"models/socialmdn_{args.test_scene}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.time()
    splits = build_splits(Path(args.data), args.test_scene, train_frac=tr["train_frac"])
    print({k: len(v) for k, v in splits.items()}, f"windows ({time.time() - t0:.1f}s)")

    model = SocialMDN(ModelConfig(**cfg["model"]))
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(args.seed)
    train_ws, val_ws = splits["train"], splits["val"]
    bs = tr["batch_size"]
    best_val, best_state, log = float("inf"), None, []

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train_ws))
        tot, n = 0.0, 0
        for b in range(0, len(perm), bs):
            obs, mask, nb, nm, fut = make_batch(train_ws, perm[b : b + bs], rng, True, tr["short_history_prob"])
            out_ = model(obs, mask, nb, nm)
            loss, _ = loss_fn(out_, fut, tr["wta_weight"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(obs)
            n += len(obs)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vn, vparts = 0.0, 0, {}
            for b in range(0, len(val_ws), 1024):
                obs, mask, nb, nm, fut = make_batch(val_ws, np.arange(b, min(b + 1024, len(val_ws))), rng, False, 0)
                l, parts = loss_fn(model(obs, mask, nb, nm), fut, tr["wta_weight"])
                vl += l.item() * len(obs)
                vn += len(obs)
                for k, v in parts.items():
                    vparts[k] = vparts.get(k, 0) + v * len(obs)
        vl /= vn
        entry = {"epoch": ep, "train_loss": tot / n, "val_loss": vl, **{f"val_{k}": v / vn for k, v in vparts.items()}}
        log.append(entry)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}), flush=True)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # --- calibration on validation (never on test), then held-out test evaluation ---
    data_hash = hashlib.sha256(b"".join(sorted(p.read_bytes() for p in Path(args.data).glob("*.txt")))).hexdigest()[:16]
    meta = {
        "model_name": "SocialMDN",
        "version": f"socialmdn-{args.test_scene}-holdout-v1",
        "trained_on": [s for s in ["eth", "hotel", "univ", "zara1", "zara2"] if s != args.test_scene],
        "held_out_test_scene": args.test_scene,
        "dt": DT,
        "obs_len": OBS_LEN,
        "pred_len": PRED_LEN,
        "units": "meters (ground plane)",
        "seed": args.seed,
        "epochs": epochs,
        "data_sha256_16": data_hash,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report = calibrate_and_report(model, splits, out, meta, r=cfg["baseline"]["kalman_r"])
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
    print(json.dumps({"model": {k: report["model"][k] for k in ("ade_4.8s", "fde_4.8s", "minade_4.8s", "minfde_4.8s")},
                      "cv": report["baselines"]["constant_velocity"]}, indent=1))
    print(f"saved -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
