"""Post-hoc calibration on the validation split, then held-out test evaluation.

Fits, on validation data only:
  * temperature T for mode logits (target: index of the closest mode),
  * a scalar multiplier on predicted standard deviations (mixture NLL),
  * the white-noise-acceleration PSD q of the Kalman-CV baseline (NLL).
Then evaluates on the held-out test scene and writes model.pt (with calibration metadata)
and report.json. Re-run on existing checkpoints without retraining:

    python training/calibrate.py --checkpoint models/socialmdn_zara1/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from motionpredict.forecasting.data import build_splits  # noqa: E402
from motionpredict.forecasting.model import load_checkpoint, save_checkpoint  # noqa: E402
from training.evaluate import evaluate_all, fit_kalman_q, fit_sigma_scale, fit_temperature, predict  # noqa: E402


def calibrate_and_report(model, splits, out_dir: Path, meta: dict, r: float = 0.01) -> dict:
    val = splits["val"]
    vp = predict(model, val)
    T, tinfo = fit_temperature(vp, val)
    s, sinfo = fit_sigma_scale(vp, val, T)
    q, qinfo = fit_kalman_q(val, r=r)
    print("calibration:", {"temperature": T, "sigma_scale": s, "kalman_q": q})
    report = evaluate_all(model, splits["test"], T, q=q, r=r, sigma_scale=s)
    report["validation_calibration"] = {**tinfo, **sinfo, **qinfo}
    m = report["model"]
    meta = {
        **meta,
        "temperature": T,
        "sigma_scale": s,
        "kalman_q": q,
        "kalman_r": r,
        "calibration": {
            "method": "temperature scaling of mode logits + scalar sigma scaling, both fitted on the temporal validation split",
            "test_ece_all_modes": m["mode_calibration_calibrated"]["ece_all_modes"],
            "test_ece_top_mode": m["mode_calibration_calibrated"]["ece_top_mode"],
            "test_brier": m["mode_calibration_calibrated"]["brier"],
            "hdr_coverage_test": m["hdr_coverage"],
        },
        "test_metrics": {k: m[k] for k in ("ade", "fde", "ade_4.8s", "fde_4.8s", "minade", "minfde", "minade_4.8s", "minfde_4.8s", "nll_per_step")},
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out_dir / "model.pt", model, meta)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/raw"))
    a = ap.parse_args()
    model, meta = load_checkpoint(a.checkpoint)
    splits = build_splits(Path(a.data), meta["held_out_test_scene"])
    rep = calibrate_and_report(model, splits, Path(a.checkpoint).parent, meta, meta.get("kalman_r", 0.01))
    m = rep["model"]
    print(json.dumps({k: m[k] for k in ("ade_4.8s", "fde_4.8s", "minade_4.8s", "minfde_4.8s", "nll_per_step")}, indent=1))
    print("ECE raw->cal", m["mode_calibration_raw"]["ece_all_modes"], m["mode_calibration_calibrated"]["ece_all_modes"])
    print("HDR", m["hdr_coverage"])


if __name__ == "__main__":
    main()
