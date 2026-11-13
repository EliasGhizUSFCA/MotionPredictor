"""Aggregate per-fold report.json files into models/benchmark.json and a markdown table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def main() -> None:
    rows, out = [], {}
    for s in SCENES:
        p = ROOT / f"models/socialmdn_{s}/report.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        m, b = r["model"], r["baselines"]
        cal = m["mode_calibration_calibrated"]
        raw = m["mode_calibration_raw"]
        out[s] = {
            "n": m["n_windows"],
            "cv_ade_fde_4.8": [b["constant_velocity"]["ade_4.8s"], b["constant_velocity"]["fde_4.8s"]],
            "kalman_ade_fde_4.8": [b["kalman_cv"]["ade_4.8s"], b["kalman_cv"]["fde_4.8s"]],
            "model_top1_ade_fde_4.8": [m["ade_4.8s"], m["fde_4.8s"]],
            "model_min5_ade_fde_4.8": [m["minade_4.8s"], m["minfde_4.8s"]],
            "nll_model": m["nll_per_step"], "nll_kalman": b["kalman_cv"]["nll_per_step"],
            "ece_raw": raw["ece_all_modes"], "ece_cal": cal["ece_all_modes"],
            "hdr90_4.8s_model": m["hdr_coverage"]["4.8s"]["0.90"], "hdr90_4.8s_kalman": b["kalman_cv"]["hdr_coverage"]["4.8s"]["0.90"],
            "latency_ms": m["latency_ms_single_window_median"],
        }
        o = out[s]
        f = lambda x: f"{x[0]:.2f}/{x[1]:.2f}"  # noqa: E731
        rows.append(
            f"| {s} | {o['n']} | {f(o['cv_ade_fde_4.8'])} | {f(o['kalman_ade_fde_4.8'])} | {f(o['model_top1_ade_fde_4.8'])} | "
            f"{f(o['model_min5_ade_fde_4.8'])} | {o['nll_kalman']:.2f} / {o['nll_model']:.2f} | {o['ece_raw']:.3f} → {o['ece_cal']:.3f} | "
            f"{o['hdr90_4.8s_kalman']:.2f} / {o['hdr90_4.8s_model']:.2f} |"
        )
    if out:
        keys = ["cv_ade_fde_4.8", "kalman_ade_fde_4.8", "model_top1_ade_fde_4.8", "model_min5_ade_fde_4.8"]
        avg = {k: [sum(v[k][i] for v in out.values()) / len(out) for i in range(2)] for k in keys}
        out["average"] = avg
        f = lambda x: f"{x[0]:.2f}/{x[1]:.2f}"  # noqa: E731
        rows.append(f"| **avg** | | {f(avg[keys[0]])} | {f(avg[keys[1]])} | {f(avg[keys[2]])} | {f(avg[keys[3]])} | | | |")
    header = (
        "| held-out scene | windows | CV ADE/FDE | Kalman-CV ADE/FDE | SocialMDN top-1 ADE/FDE | SocialMDN min-of-5 ADE/FDE | "
        "NLL/step Kalman / MDN | mode ECE raw → calibrated | 90% HDR coverage @4.8s Kalman / MDN |\n|---|---|---|---|---|---|---|---|---|"
    )
    md = header + "\n" + "\n".join(rows)
    (ROOT / "models/benchmark.json").write_text(json.dumps(out, indent=2))
    (ROOT / "models/benchmark.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
