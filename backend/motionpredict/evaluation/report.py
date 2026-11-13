"""Aggregation, CSV/JSON export and a self-contained HTML evaluation report."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METRICS = ["ade_top1", "fde_top1", "min_ade", "min_fde", "w_ade", "w_fde", "nll_final"]
BH_METRICS = ["ade_top1_bh", "fde_top1_bh", "min_ade_bh", "min_fde_bh", "w_ade_bh", "w_fde_bh"]


def load_predictions(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "predictions.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def rows_from_predictions(preds: list[dict]) -> list[dict]:
    """One row per (prediction, horizon, model); excluded samples have included=False."""
    rows = []
    for p in preds:
        for h, ev in p["evaluations"].items():
            base = {"prediction_id": p["id"], "track_id": p["track_id"], "object_class": p["object_class"], "t": p["t"],
                    "horizon": float(h), "coord": p["coord"], "start_x": p["start"][0], "start_y": p["start"][1],
                    "start_px_x": p["start_px"][0], "start_px_y": p["start_px"][1], "body_px": p["body_px"],
                    "tracker_confidence": p["track_quality"]["tracker_confidence"], "history_steps": p["observation_window"]["steps"],
                    "included": ev["included"], "exclusion_reasons": ";".join(ev.get("exclusion_reasons", []))}
            for m, mo in p["models"].items():
                r = {**base, "model": m, "model_version": mo["model_version"], "probability_type": mo["probability_type"],
                     "calibration_status": mo["calibration_status"], "latency_ms": mo["latency_ms"]}
                if ev["included"]:
                    s = ev["scores"][m]
                    r.update({k: s.get(k) for k in METRICS + BH_METRICS + ["miss_top1", "miss_k", "in_hdr50", "in_hdr90", "top_id",
                                                                         "top_prob", "best_id", "k"]})
                    r["mode_probs"] = s.get("mode_probs")
                rows.append(r)
    return rows


def _stats(v: list[float]) -> dict | None:
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return None
    a = np.asarray(v)
    med = float(np.median(a))
    return {"mean": float(a.mean()), "median": med, "p50": med, "p90": float(np.percentile(a, 90)), "n": len(a)}


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    d = a - b
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_pairs": int(len(d))}


def reliability(rows: list[dict], n_bins: int = 10) -> dict:
    """Mode-probability reliability: event = 'this mode turned out closest to the actual path'."""
    conf, hit = [], []
    for r in rows:
        mp = r.get("mode_probs")
        if not mp or any(v is None for v in mp.values()):
            continue
        for k, p in mp.items():
            conf.append(p)
            hit.append(1.0 if k == r["best_id"] else 0.0)
    if not conf:
        return {"available": False}
    conf, hit = np.asarray(conf), np.asarray(hit)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(conf[m].mean()), "frequency": float(hit[m].mean())})
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return {"available": True, "ece": float(ece), "bins": bins, "n_mode_predictions": int(len(conf)),
            "brier_like": float(((conf - hit) ** 2).mean())}


def heatmaps(rows: list[dict], preds: list[dict], coord: str, frame_wh: tuple[int, int] | None, nx: int = 12, ny: int = 8) -> dict:
    key = ("start_px_x", "start_px_y")
    if frame_wh and frame_wh[0]:
        x0, x1, y0, y1 = 0, frame_wh[0], 0, frame_wh[1]
    else:
        xs = [r[key[0]] for r in rows] or [0, 1]
        ys = [r[key[1]] for r in rows] or [0, 1]
        x0, x1, y0, y1 = min(xs), max(xs) + 1e-6, min(ys), max(ys) + 1e-6

    def cell(r):
        i = int(np.clip((r[key[0]] - x0) / (x1 - x0) * nx, 0, nx - 1))
        j = int(np.clip((r[key[1]] - y0) / (y1 - y0) * ny, 0, ny - 1))
        return i, j

    err = defaultdict(list)
    excl = defaultdict(lambda: [0, 0])
    for r in rows:
        i, j = cell(r)
        excl[(r["model"], r["horizon"], i, j)][1] += 1
        if r["included"]:
            v = r["fde_top1_bh"] if coord == "image_pixels" else r["fde_top1"]
            err[(r["model"], r["horizon"], i, j)].append(v)
        else:
            excl[(r["model"], r["horizon"], i, j)][0] += 1
    # direction changes observed in the actual future (heading change first vs second half, degrees)
    turn = defaultdict(list)
    for p in preds:
        a = np.asarray(p["actual_future"], float)
        if len(a) >= 5:
            half = len(a) // 2
            v1, v2 = a[half, 1:] - a[0, 1:], a[-1, 1:] - a[half, 1:]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
                i, j = cell({key[0]: p["start_px"][0], key[1]: p["start_px"][1]})
                turn[(i, j)].append(ang)
    cells = []
    for (m, h, i, j), (ne, nt) in excl.items():
        e = err.get((m, h, i, j), [])
        cells.append({"model": m, "horizon": h, "i": i, "j": j, "n": len(e), "n_total": nt,
                      "mean_fde": float(np.mean(e)) if e else None, "exclusion_rate": ne / nt if nt else None,
                      "mean_turn_deg": float(np.mean(turn[(i, j)])) if turn.get((i, j)) else None})
    return {"grid": {"nx": nx, "ny": ny, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "space": "image pixels (start position)"},
            "unit": "body heights" if coord == "image_pixels" else "m", "min_samples_to_display": 3, "cells": cells}


def summarise(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    preds = load_predictions(run_dir)
    rows = rows_from_predictions(preds)
    coord = meta["coordinate_system"]
    unit = "m" if coord == "ground_plane_meters" else "px"
    models = meta["eval_config"]["models"]
    horizons = meta["eval_config"]["horizons"]
    flags = json.loads((run_dir / "flags.json").read_text()) if (run_dir / "flags.json").exists() else []
    flagged_tracks = {f["track_id"] for f in flags if f.get("kind") == "inaccurate_track" and f.get("track_id") is not None}
    # reviewer-flagged inaccurate tracks are removed once, before EVERY aggregate below
    n_flagged_rows = Counter(r["horizon"] for r in rows if r["track_id"] in flagged_tracks and r["model"] == models[0])
    all_rows = rows  # the CSV export keeps every row (flagged ones included) for inspection
    rows = [r for r in rows if r["track_id"] not in flagged_tracks]
    preds = [p for p in preds if p["track_id"] not in flagged_tracks]
    ff = meta.get("forecast_failures") or {}
    n_elig = ff.get("n_eligible") or 0
    fail_by_model = ff.get("by_model") or {}

    excl = defaultdict(Counter)
    per = {}
    for h in horizons:
        hr = [r for r in rows if r["horizon"] == h and r["model"] == models[0]]
        excl[str(h)]["included"] = sum(r["included"] for r in hr)
        for r in hr:
            for reason in filter(None, r["exclusion_reasons"].split(";")):
                excl[str(h)][reason] += 1
        if ff.get("count"):
            excl[str(h)]["forecast_failed"] = ff["count"]  # eligible tracks with no forecast from some model (no sample)
        if n_flagged_rows.get(h):
            excl[str(h)]["flagged_inaccurate_track"] = n_flagged_rows[h]
        for m in models:
            mr = [r for r in rows if r["horizon"] == h and r["model"] == m and r["included"]]
            d = {"n": len(mr)}
            for k in METRICS:
                d[k] = _stats([r.get(k) for r in mr])
            if coord == "image_pixels":
                for k in BH_METRICS:
                    d[k] = _stats([r.get(k) for r in mr])
            # share of eligible (confirmed, detected) tracks at prediction times this model failed to forecast
            d["forecast_failure_rate"] = (fail_by_model.get(m, 0) / n_elig) if n_elig else None
            if mr:
                d["miss_rate_top1"] = float(np.mean([r["miss_top1"] for r in mr]))
                d["miss_rate_k"] = float(np.mean([r["miss_k"] for r in mr]))
                d["coverage50"] = float(np.mean([r["in_hdr50"] for r in mr]))
                d["coverage90"] = float(np.mean([r["in_hdr90"] for r in mr]))
                d["latency_ms"] = float(np.mean([r["latency_ms"] for r in mr]))
                d["k"] = mr[0]["k"]
                d["probability_type"] = mr[0]["probability_type"]
                d["calibration_status"] = mr[0]["calibration_status"]
                d["reliability"] = reliability(mr)
            per[f"{m}@{h}"] = d

    paired = {}
    if len(models) == 2:
        for h in horizons:
            a = {r["prediction_id"]: r for r in rows if r["horizon"] == h and r["model"] == models[1] and r["included"]}
            b = {r["prediction_id"]: r for r in rows if r["horizon"] == h and r["model"] == models[0] and r["included"]}
            ids = sorted(set(a) & set(b))
            if len(ids) >= 5:
                k = "_bh" if coord == "image_pixels" else ""
                paired[str(h)] = {
                    "comparison": f"{models[1]} minus {models[0]} (negative = {models[1]} better)",
                    "ade_top1": _bootstrap_diff(np.array([a[i]["ade_top1" + k] for i in ids]), np.array([b[i]["ade_top1" + k] for i in ids])),
                    "fde_top1": _bootstrap_diff(np.array([a[i]["fde_top1" + k] for i in ids]), np.array([b[i]["fde_top1" + k] for i in ids])),
                    "unit": "body heights" if coord == "image_pixels" else "m",
                }

    sel_h = 3.0 if 3.0 in horizons else horizons[-1]
    ref_model = "learned" if "learned" in models else models[0]
    ek = "fde_top1_bh" if coord == "image_pixels" else "fde_top1"
    scored = sorted([r for r in rows if r["horizon"] == sel_h and r["model"] == ref_model and r["included"]], key=lambda r: -r[ek])
    by_track = defaultdict(list)
    for r in rows:
        if r["included"] and r["horizon"] == sel_h:
            by_track[(r["model"], r["track_id"])].append(r[ek])
    frames = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text().splitlines()] if (run_dir / "frames.jsonl").exists() else []
    stride = max(1, len(frames) // 1500)
    vid = meta.get("video", {})
    summary = {
        "run_id": meta["run_id"],
        "coordinate_system": coord,
        "unit": unit,
        "normalised_unit": "body heights" if coord == "image_pixels" else None,
        "ground_truth": meta["ground_truth"],
        "models": models,
        "horizons": horizons,
        "n_predictions": len(preds),
        "inclusion": {h: dict(c) for h, c in excl.items()},
        "flagged_tracks_excluded": sorted(flagged_tracks),
        "forecast_failures": {"count": int(ff.get("count", 0)), "by_reason": dict(ff.get("by_reason") or {}),
                              "by_model": dict(fail_by_model), "n_eligible": int(n_elig)},
        "scene_in_active_model_training": (meta.get("source") or {}).get("scene_in_active_model_training"),
        "per_model_horizon": per,
        "paired_comparison": paired,
        "per_track": [{"model": m, "track_id": t, "n": len(v), "mean_fde": float(np.mean(v))} for (m, t), v in by_track.items()],
        "failure_cases": [{k: r[k] for k in ("prediction_id", "track_id", "t", "horizon", ek, "top_id", "best_id")} for r in scored[:50]],
        "best_cases": [{k: r[k] for k in ("prediction_id", "track_id", "t", "horizon", ek, "top_id", "best_id")} for r in scored[::-1][:10]],
        "selected_horizon_for_rankings": sel_h,
        "ranking_metric": ek,
        "tracking_over_time": frames[::stride],
        "heatmap": heatmaps(rows, preds, coord, (vid.get("width"), vid.get("height"))),
        "metric_definitions": {
            "ade_top1": "mean distance of the highest-probability (or only) path to the actual path",
            "fde_top1": "distance at the horizon for the highest-probability path",
            "min_ade/min_fde": "best of K hypotheses chosen with hindsight (oracle) - not a model choice",
            "w_ade/w_fde": "probability-weighted ADE / FDE (learned model only)",
            "forecast_failure_rate": "eligible tracks (confirmed and detected at the prediction time) for which this model "
                                     "returned no forecast or raised; such tracks yield no sample for any model",
            "miss_rate": f"FDE > {'2.0 m' if unit == 'm' else '1 body height'}",
            "nll_final": "negative log-likelihood of the actual final position (lower is better; same coordinate system only)",
            "coverage50/90": "fraction of actual final positions inside the 50%/90% predicted region",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_csv(run_dir, all_rows)
    (run_dir / "report.html").write_text(render_report(meta, summary, preds))
    return summary


def write_csv(run_dir: Path, rows: list[dict]) -> None:
    cols = ["prediction_id", "track_id", "object_class", "t", "horizon", "model", "model_version", "coord", "included",
            "exclusion_reasons", "ade_top1", "fde_top1", "min_ade", "min_fde", "w_ade", "w_fde", "ade_top1_bh", "fde_top1_bh", "min_ade_bh",
            "min_fde_bh", "w_ade_bh", "w_fde_bh", "miss_top1", "miss_k", "nll_final", "in_hdr50", "in_hdr90", "top_id", "top_prob", "best_id", "k",
            "probability_type", "calibration_status", "latency_ms", "tracker_confidence", "history_steps", "start_x", "start_y", "body_px"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    (run_dir / "predictions.csv").write_text(buf.getvalue())


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _example_svg(p: dict, model: str, h: str, unit: str) -> str:
    mo = p["models"][model]
    act = np.asarray(p["actual_future"], float)
    paths = [np.asarray(t["points"])[np.asarray(t["times"]) <= float(h) + 1e-6] for t in mo["trajectories"][:3]]
    allp = np.vstack([act[:, 1:]] + paths)
    lo, hi = allp.min(0), allp.max(0)
    span = max(float((hi - lo).max()), 1e-6)
    S = 180
    f = lambda q: ((q[:, 0] - lo[0]) / span * (S - 20) + 10, (q[:, 1] - lo[1]) / span * (S - 20) + 10)  # noqa: E731
    flip = unit == "m"
    out = [f'<svg width="{S}" height="{S}" viewBox="0 0 {S} {S}" style="background:#0f141b;border:1px solid #273142">']
    cols = ["#22c55e", "#eab308", "#f43f5e"]
    for c, q in zip(cols, paths):
        x, y = f(q)
        y = S - y if flip else y
        out.append(f'<polyline fill="none" stroke="{c}" stroke-dasharray="4 3" stroke-width="2" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))}"/>')
    a = act[act[:, 0] <= float(h) + 1e-6, 1:]
    x, y = f(a)
    y = S - y if flip else y
    out.append(f'<polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in zip(x, y))}"/>')
    out.append(f'<circle cx="{x[0]:.1f}" cy="{y[0]:.1f}" r="4" fill="#fff"/></svg>')
    return "".join(out)


def render_report(meta: dict, s: dict, preds: list[dict]) -> str:
    e = html.escape
    unit = s["unit"]
    nu = " (body heights)" if s["normalised_unit"] else f" ({unit})"
    sfx = "_bh" if s["normalised_unit"] else ""
    rows = []
    for key, d in s["per_model_horizon"].items():
        m, h = key.split("@")
        g = lambda k: (f"{d[k]['mean']:.3f} / {d[k]['median']:.3f} / {d[k]['p90']:.3f}" if d.get(k) else "–")  # noqa: E731
        fr = d.get("forecast_failure_rate")
        rows.append(f"<tr><td>{e(m)}</td><td>{h}s</td><td>{d['n']}</td><td>{g('ade_top1' + sfx)}</td><td>{g('fde_top1' + sfx)}</td>"
                    f"<td>{g('min_ade' + sfx)}</td><td>{g('min_fde' + sfx)}</td><td>{g('w_ade' + sfx)}</td><td>{g('w_fde' + sfx)}</td>"
                    f"<td>{d.get('miss_rate_top1', float('nan')):.2f}</td>"
                    f"<td>{d.get('coverage50', float('nan')):.2f} / {d.get('coverage90', float('nan')):.2f}</td>"
                    f"<td>{g('nll_final')}</td><td>{d.get('latency_ms', float('nan')):.2f}</td>"
                    f"<td>{'–' if fr is None else f'{fr:.3f}'}</td><td>{e(str(d.get('probability_type', '')))}</td></tr>")
    paired = "".join(f"<tr><td>{h}s</td><td>{v['ade_top1']['mean_diff']:.3f} [{v['ade_top1']['ci95'][0]:.3f}, {v['ade_top1']['ci95'][1]:.3f}]</td>"
                     f"<td>{v['fde_top1']['mean_diff']:.3f} [{v['fde_top1']['ci95'][0]:.3f}, {v['fde_top1']['ci95'][1]:.3f}]</td><td>{v['ade_top1']['n_pairs']}</td></tr>"
                     for h, v in s["paired_comparison"].items())
    incl = "".join(f"<tr><td>{h}s</td><td>{e(json.dumps(c))}</td></tr>" for h, c in s["inclusion"].items())
    rel = ""
    for key, d in s["per_model_horizon"].items():
        r = d.get("reliability") or {}
        if r.get("available"):
            rel += f"<h4>{e(key)} - ECE {r['ece']:.3f} over {r['n_mode_predictions']} mode predictions</h4><table><tr><th>bin</th><th>n</th><th>mean prob</th><th>observed freq</th></tr>"
            rel += "".join(f"<tr><td>{b['lo']:.1f}-{b['hi']:.1f}</td><td>{b['n']}</td><td>{b['confidence']:.3f}</td><td>{b['frequency']:.3f}</td></tr>" for b in r["bins"])
            rel += "</table>"
    byid = {p["id"]: p for p in preds}
    model = "learned" if "learned" in s["models"] else s["models"][0]
    ex = ""
    for title, lst in (("Largest errors", s["failure_cases"][:4]), ("Smallest errors", s["best_cases"][:4])):
        ex += f"<h4>{title} ({e(model)}, {s['selected_horizon_for_rankings']}s)</h4><div class=ex>"
        for c in lst:
            p = byid.get(c["prediction_id"])
            if p:
                ex += f"<figure>{_example_svg(p, model, str(c['horizon']), unit)}<figcaption>track {c['track_id']} @ {c['t']:.1f}s<br>FDE {c[s['ranking_metric']]:.2f}</figcaption></figure>"
        ex += "</div>"
    import torch

    src = meta.get("source") or {}
    is_scene = src.get("type") == "dataset_scene"
    limits = []
    if is_scene:
        limits.append("Ground truth is human-annotated ETH/UCY positions (not tracker output); the input frames are a "
                      "synthetic replay of those annotations, so detection and tracking are idealised compared with real video.")
    else:
        limits.append("Ground truth from the tracker inherits detection noise, occlusion gaps and ID switches (pseudo ground truth).")
    limits.append(f"Quality gates: <code>{e(json.dumps(meta['eval_config']['quality_rules']))}</code>; a quality event only "
                  "excludes horizons whose ground truth it affects.")
    if s["coordinate_system"] == "ground_plane_meters":
        limits.append("Errors are in ground-plane meters via " + ("the dataset's homography." if is_scene else
                      "the user-supplied camera calibration; calibration error adds to forecast error, especially far from the reference points."))
    else:
        limits.append("Uncalibrated video: errors are in image pixels, distorted by perspective; body-height normalisation reduces but does not remove this.")
    limits.append("The learned model was trained on pedestrian data (ETH/UCY); other motion (e.g. sports) is out of distribution.")
    limits_html = "".join(f"<li>{x}</li>" for x in limits)
    train_warn = ""
    if src.get("scene_in_active_model_training"):
        train_warn = (f"<div class=warn><b>Warning:</b> scene <code>{e(str(src.get('scene')))}</code> was part of the learned model's "
                      "training data; its results here are optimistic and are not a held-out evaluation.</div>")
    elif is_scene and src.get("scene_in_active_model_training") is None and "learned" in s["models"]:
        train_warn = "<div class=warn>It is unknown whether this scene was in the learned model's training data.</div>"
    ff = s.get("forecast_failures") or {}
    fail_html = (f"<p>Forecast failures (eligible tracks with no forecast from some model; no sample for any model): "
                 f"{ff.get('count', 0)} of {ff.get('n_eligible', 0)} - {e(json.dumps(ff.get('by_reason', {})))}</p>")

    return f"""<!doctype html><html><head><meta charset=utf-8><title>MotionPredict evaluation {e(s['run_id'])}</title>
<style>body{{background:#0b0f14;color:#d6dde6;font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}
h1,h2,h3{{color:#fff}}table{{border-collapse:collapse;margin:8px 0 16px;width:100%}}td,th{{border:1px solid #273142;padding:4px 8px;text-align:left}}
th{{background:#121924}}.warn{{border-left:3px solid #f59e0b;padding:6px 12px;background:#1a1408}}.ex{{display:flex;gap:12px;flex-wrap:wrap}}
figure{{margin:0}}figcaption{{font-size:12px;color:#94a3b8}}code{{color:#93c5fd}}</style></head><body>
<h1>MotionPredict AI - prediction vs. reality report</h1>
<p>Run <code>{e(s['run_id'])}</code> · created {e(meta['created'])} · source: {e(meta['source']['type'])} {e(str(meta['source'].get('filename') or meta['source'].get('scene') or ''))}</p>
<div class=warn><b>Ground truth:</b> {e(meta['ground_truth'])}. <b>Coordinates:</b> {e(s['coordinate_system'])}.
{"Pixel errors are also normalised by the person's bounding-box height (body heights) so near and far people are comparable; they are not metric." if s['normalised_unit'] else ""}
Best-of-K (minADE/minFDE) uses hindsight to pick the closest hypothesis and does not mean the model selected it in advance.
Forecasts are estimates; this system is not suitable for safety-critical use.</div>
{train_warn}
<h2>Video / data</h2><pre>{e(json.dumps(meta.get('video', {}), indent=1))}</pre>
<p>Tags: {e(json.dumps(meta.get('tags', {})))}</p>
<h2>Configuration</h2><pre>{e(json.dumps({'eval': meta['eval_config'], 'models': meta['models'], 'detector': meta['detector']}, indent=1))}</pre>
<h2>Sample counts and exclusions</h2><table><tr><th>horizon</th><th>counts</th></tr>{incl}</table>{fail_html}
{f"<p>Excluded as inaccurate tracks (from every statistic): {e(', '.join(f'#{t}' for t in s['flagged_tracks_excluded']))}</p>" if s.get('flagged_tracks_excluded') else ""}
<h2>Accuracy (mean / median / p90){e(nu)}</h2>
<table><tr><th>model</th><th>horizon</th><th>n</th><th>ADE top-1</th><th>FDE top-1</th><th>minADE@K</th><th>minFDE@K</th><th>wADE</th><th>wFDE</th><th>miss rate</th><th>coverage 50/90%</th><th>NLL final</th><th>latency ms</th><th>failure rate</th><th>probabilities</th></tr>{''.join(rows)}</table>
<h2>Paired comparison on identical samples (bootstrap 95% CI)</h2><table><tr><th>horizon</th><th>ΔADE</th><th>ΔFDE</th><th>pairs</th></tr>{paired}</table>
<h2>Calibration diagnostics</h2>{rel or "<p>No probabilistic mode outputs in this run.</p>"}
<h2>Examples</h2><p>Dashed: predicted (green = highest probability, yellow, red). Blue: actual. White: start.</p>{ex}
<h2>Limitations</h2><ul>{limits_html}</ul>
<h2>Reproducibility</h2><pre>{e(json.dumps({'git': _git_rev(), 'python': platform.python_version(), 'torch': torch.__version__, 'source_sha256': meta['source'].get('sha256'), 'pipeline_config': meta['pipeline_config']}, indent=1))}</pre>
</body></html>"""
