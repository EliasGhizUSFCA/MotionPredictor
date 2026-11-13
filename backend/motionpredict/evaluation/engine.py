"""Whole-recording prediction-vs-reality evaluation.

Causality guarantee
    Frames are processed strictly in timestamp order by the same ``Session`` used live. At a
    prediction time t the forecasts of every model are computed from the session state *at that
    moment* and written to disk immediately (``predictions.jsonl`` is append-only). The
    observed future is attached afterwards, as later frames arrive. Each record stores
    ``causality.latest_frame_t_used`` which tests assert equals the prediction time.

Ground truth
    * ``video``           - the tracker's later observed (detection-matched) footpoints for the
                            same track ID. This is *tracker-derived pseudo ground truth*: it
                            inherits detector noise and possible ID switches, so samples are
                            quality-gated (see ``QUALITY_RULES``) and exclusions are reported.
    * ``dataset_scene``   - human-annotated ETH/UCY positions (true ground truth), used to
                            validate the engine itself against the offline benchmark.

Both models are evaluated on identical samples (same track, same time, same history, same
future) and the paired differences are bootstrapped.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calibration import fit_calibration
from ..forecasting.constants import DT
from ..forecasting.service import Forecaster
from ..pipeline import Session
from .scoring import score_forecast

UI_HORIZONS = [0.5, 1.0, 2.0, 3.0, 5.0]
QUALITY_RULES = {
    "max_gt_gap_s": 0.6,  # longest allowed gap between observed future positions
    "min_tracker_confidence": 0.3,  # at prediction time
    "max_speed_m_s": 7.0,  # faster jumps in the observed future => likely ID switch
    "max_speed_bh_s": 5.0,  # same, in body heights / s (pixel mode)
    "camera_motion_fraction": 0.3,  # fraction of future frames flagged as camera moving
    "cut_threshold": 40.0,  # mean abs grey-level difference (0-255) between frames => cut
}
EVAL_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "evaluations"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _r(v, d):
    return None if v is None or not math.isfinite(v) else round(float(v), d)


def compact(f) -> dict:
    """Stored form of a forecast: rounded, without the derived band polygons (rebuilt from sigma in the UI)."""
    out = f.model_dump(include={"prediction_source", "model_version", "probability_type", "calibration_status", "history_steps_used"})
    out["trajectories"] = [{
        "id": t.id, "rank": t.rank, "probability": t.probability, "times": t.times,
        "points": [[_r(x, 4), _r(y, 4)] for x, y in t.points],
        "image_points": [[_r(x, 1), _r(y, 1)] for x, y in (t.image_points or [])],
        "sigma": [[_r(a, 4), _r(b, 4), _r(c, 3)] for a, b, c in t.sigma],
    } for t in f.trajectories]
    return out


class EvaluationRun:
    def __init__(self, root: Path, run_id: str | None = None):
        self.id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = Path(root) / self.id
        self.status = "created"
        self.progress = 0.0
        self.error: str | None = None
        self.cancel = threading.Event()

    # ------------------------------------------------------------ public
    def run(self, cfg: dict, detector, forecaster: Forecaster, video_path: Path | None = None, scene: str | None = None,
            raw_dir: Path | None = None, keep_video: bool = False) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = "running"
        try:
            self._run(cfg, detector, forecaster, video_path, scene, raw_dir)
            self.status = "cancelled" if self.cancel.is_set() else "done"
        except Exception as e:
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            (self.dir / "error.txt").write_text(traceback.format_exc())
        finally:
            if video_path is not None and not keep_video and Path(video_path).exists():
                Path(video_path).unlink()  # privacy: uploaded footage is deleted after analysis
            self._write_status()

    def _write_status(self) -> None:
        (self.dir / "status.json").write_text(json.dumps({"status": self.status, "progress": self.progress, "error": self.error}))

    # ----------------------------------------------------------- frames
    def _frames_video(self, path: Path, process_fps: float):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("could not open video (unsupported codec/container?)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / process_fps)))
        self.video_info = {"fps": fps, "frame_count": n, "duration_s": n / fps if n else None, "process_every_nth": step,
                           "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield i, i / fps, frame, None
                if n:
                    self.progress = i / n
            i += 1
        cap.release()

    def _frames_scene(self, det, max_s: float | None):
        rp = self._replay
        dur = rp.duration if max_s is None else min(rp.duration, max_s)
        i = 0
        while rp.t + rp.tick_s <= dur:
            frame, t, mapping = rp.step(det)
            i += 1
            self.progress = t / dur
            yield i, t, frame, mapping

    # -------------------------------------------------------------- core
    def _run(self, cfg: dict, detector, forecaster: Forecaster, video_path, scene, raw_dir) -> None:
        interval = float(cfg.get("interval_s", 1.0))
        horizons = sorted(float(h) for h in cfg.get("horizons", UI_HORIZONS))
        hmax = max(horizons)
        models = cfg.get("models", ["baseline", "learned"])
        if "learned" in models and not forecaster.available:
            models = [m for m in models if m != "learned"]
            cfg["notes_models"] = f"learned model unavailable: {forecaster.load_error}"
        is_scene = scene is not None
        if is_scene:
            from ..demo import ReplayDetector, SceneReplay

            detector = ReplayDetector()
            self._replay = SceneReplay(raw_dir, scene, tick_s=0.1)
            d = self._replay.duration if cfg.get("max_duration_s") is None else min(self._replay.duration, cfg["max_duration_s"])
            self.video_info = {"fps": 10.0, "frame_count": int(d / 0.1), "duration_s": d, "width": self._replay.W,
                               "height": self._replay.H, "process_every_nth": 1, "dataset_file": self._replay.file}
        sess = Session(detector, forecaster)
        pcfg = sess.config.model_copy(update={k: v for k, v in cfg.get("pipeline", {}).items() if k in sess.config.model_fields})
        pcfg = pcfg.model_copy(update={"forecaster": "baseline", "history_s": max(pcfg.history_s, 6.0)})
        if is_scene:
            pcfg = pcfg.model_copy(update={"pose": False, "camera_motion_check": False, "det_conf": 0.5})
        sess.apply_config(pcfg)
        frames = self._frames_scene(detector, cfg.get("max_duration_s")) if is_scene else \
            self._frames_video(Path(video_path), float(cfg.get("process_fps", 10.0)))
        if is_scene:
            sess.calibration = self._replay.calibration
        elif cfg.get("calibration"):
            c = cfg["calibration"]
            sess.calibration = fit_calibration(c["image_points"], c["world_points"], c.get("camera_height_m"), c.get("reference"))

        meta: dict[str, Any] = {
            "run_id": self.id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {"type": "dataset_scene" if is_scene else "video", "scene": scene,
                       "filename": cfg.get("filename"), "sha256": _sha256(Path(video_path)) if video_path else None},
            "ground_truth": "human annotations (ETH/UCY)" if is_scene else "tracker-derived pseudo ground truth (later detections of the same track)",
            "tags": cfg.get("tags", {}),
            "eval_config": {"interval_s": interval, "horizons": horizons, "models": models, "k_learned": 5,
                            "process_fps": cfg.get("process_fps", 10.0), "quality_rules": QUALITY_RULES},
            "pipeline_config": pcfg.model_dump(),
            "detector": getattr(detector, "weights", None),
            "models": {"baseline": {"version": "kalman-cv/ca-v1", "kalman_q_m": forecaster.kalman_q},
                       "learned": {"version": forecaster.meta.get("version"), "temperature": forecaster.meta.get("temperature"),
                                   "sigma_scale": forecaster.meta.get("sigma_scale"),
                                   "probability_calibration_validated": forecaster.validated} if forecaster.available else None},
            "object_classes": ["person"],
            "coordinate_system": "ground_plane_meters" if sess.calibration else "image_pixels",
            "calibration": sess.calibration.to_dict() if sess.calibration else None,
        }
        pred_f = open(self.dir / "predictions.jsonl", "w")
        frames_f = open(self.dir / "frames.jsonl", "w")
        pending: list[dict] = []  # forecasts awaiting their future
        obs_px: dict[int, list] = defaultdict(list)  # observed footpoints per track: [t, x_px, y_px, x_w, y_w]
        next_pred_t = None
        prev_small = None
        n_pred = 0
        t_last = 0.0
        for fidx, t, frame, mapping in frames:
            if self.cancel.is_set():
                break
            t_last = t
            small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            cut = prev_small is not None and float(np.abs(small - prev_small).mean()) > QUALITY_RULES["cut_threshold"]
            prev_small = small
            if cut:  # a hard cut breaks every track; start fresh to avoid cross-shot associations
                sess.reset()
                for p in pending:
                    p["flags"].add("camera_cut")
            res = sess.process(frame, fidx, t, time.time() * 1000, None, "dataset_replay" if is_scene else "server", 0.0, 0)
            cam_moving = bool(res.camera_motion and res.camera_motion.get("camera_moving"))
            frames_f.write(json.dumps({"t": round(t, 3), "frame": fidx, "tracks": sum(x.state == "confirmed" for x in res.tracks),
                                       "detections": res.detections, "camera_moving": cam_moving, "cut": cut,
                                       "mean_tracker_conf": round(float(np.mean([x.tracker_confidence for x in res.tracks])), 3) if res.tracks else None,
                                       "processing_ms": res.processing_ms}) + "\n")

            # ground-truth observations at this frame
            matched = {}
            if is_scene:
                for tr in sess.tracker.tracks:
                    if tr.matched_this_frame and tr.last_det_box is not None:
                        for pid, box in mapping.items():
                            if np.allclose(tr.last_det_box, box):
                                gx, gy = self._replay.positions_at(t)[pid]
                                u, v = self._replay.to_px(np.array([gx, gy]))[0]
                                matched[tr.track_id] = (u, v, gx, gy, pid)
            else:
                for o in sess.last_outs:
                    tr = o["tr"]
                    if tr.matched_this_frame:
                        u, v = o["foot"]
                        g = (None, None)
                        if sess.calibration is not None:
                            W, H = sess.last_wh
                            gg = sess.calibration.to_ground(np.array([[u / W, v / H]]))[0]
                            g = (float(gg[0]), float(gg[1]))
                        matched[tr.track_id] = (u, v, g[0], g[1], None)
            for tid, (u, v, gx, gy, _) in matched.items():
                obs_px[tid].append([round(t, 3), round(float(u), 1), round(float(v), 1),
                                    None if gx is None else round(gx, 3), None if gy is None else round(gy, 3)])
            for p in pending:
                dtp = t - p["t"]
                if dtp <= 1e-9:
                    continue
                if cam_moving:
                    p["cam_frames"] += 1
                p["future_frames"] += 1
                if p["track_id"] in matched:
                    u, v, gx, gy, pid = matched[p["track_id"]]
                    if is_scene and p.get("pid") is not None and pid != p["pid"]:
                        p["flags"].add("id_switch")
                    w = (u, v) if p["coord"] == "image_pixels" else (gx, gy)
                    p["gt"].append([round(dtp, 3), float(w[0]), float(w[1])])
                elif all(x.track_id != p["track_id"] for x in sess.tracker.tracks):
                    p["flags"].add("track_lost")

            # finalise pending forecasts whose longest horizon has elapsed
            still = []
            for p in pending:
                if t - p["t"] >= p["hmax"] + 0.05 or "track_lost" in p["flags"] or "camera_cut" in p["flags"]:
                    pred_f.write(json.dumps(self._finalise(p)) + "\n")
                else:
                    still.append(p)
            pending = still

            # new forecasts at this time (strictly from current state)
            if next_pred_t is None:
                next_pred_t = t
            if t + 1e-6 >= next_pred_t:
                next_pred_t += interval
                elig = {o["id"] for o in sess.last_outs if o["tr"].state == "confirmed" and o["tr"].matched_this_frame}
                outs_by_model, lat = {}, {}
                for m in models:
                    t0 = time.perf_counter()
                    res_m, _ = sess.forecast_now(m, hmax, 5, elig)
                    lat[m] = (time.perf_counter() - t0) * 1000 / max(len(res_m), 1)
                    outs_by_model[m] = res_m
                for o in sess.last_outs:
                    tid = o["id"]
                    if tid not in elig or not all(tid in outs_by_model[m] for m in models):
                        continue
                    kin = o["kin"]
                    obs, nv = kin.resampled()
                    tr = o["tr"]
                    rec = {
                        "id": f"{self.id}-{n_pred:06d}", "track_id": tid, "object_class": "person", "t": round(t, 3), "frame": fidx,
                        "coord": sess.last_coord, "hmax": hmax, "horizons": horizons,
                        "body_px": float(np.median(o["st"]["hpx"])),
                        "start_px": o["foot"], "start": [float(kin.kf.x[0]), float(kin.kf.x[1])],
                        "observation_window": {"dt_s": DT, "steps": int(nv), "positions": np.round(obs[-nv:], 3).tolist()},
                        "history_px": o["hist_px"][-60:],
                        "track_quality": {"age_s": round(t - tr.start_t, 2), "hits": tr.hits,
                                          "tracker_confidence": round(tr.confidence(t, sess.tracker.track_buffer_s), 3),
                                          "det_confidence": round(tr.det_conf, 3)},
                        "camera_moving_at_t": cam_moving,
                        "causality": {"latest_frame_t_used": round(t, 3), "forecast_generated_before_future_frames": True},
                        "models": {m: {**compact(outs_by_model[m][tid]), "latency_ms": round(lat[m], 3)} for m in models},
                        "gt": [[0.0, *(o["foot"] if sess.last_coord == "image_pixels" else [float(kin.kf.x[0]), float(kin.kf.x[1])])]],
                        "flags": set(), "future_frames": 0, "cam_frames": 0,
                    }
                    if is_scene and tid in matched:
                        rec["pid"] = matched[tid][4]
                    if rec["track_quality"]["tracker_confidence"] < QUALITY_RULES["min_tracker_confidence"]:
                        rec["flags"].add("low_tracker_confidence")
                    pending.append(rec)
                    n_pred += 1
        for p in pending:
            p["flags"].add("recording_ended")
            pred_f.write(json.dumps(self._finalise(p)) + "\n")
        pred_f.close()
        frames_f.close()
        meta["video"] = self.video_info
        meta["processed_until_s"] = t_last
        meta["n_predictions"] = n_pred
        (self.dir / "run.json").write_text(json.dumps(meta, indent=2))
        (self.dir / "tracks.json").write_text(json.dumps({str(k): v for k, v in obs_px.items()}))
        if not (self.dir / "flags.json").exists():
            (self.dir / "flags.json").write_text("[]")
        self.progress = 1.0
        from .report import summarise

        summarise(self.dir)

    def _finalise(self, p: dict) -> dict:
        """Quality-gate the observed future and score every model at every horizon."""
        gt = np.asarray(p["gt"], float)
        flags = set(p["flags"])
        if p["future_frames"] and p["cam_frames"] / p["future_frames"] > QUALITY_RULES["camera_motion_fraction"]:
            flags.add("camera_motion")
        if len(gt) >= 2:
            d = np.linalg.norm(np.diff(gt[:, 1:], axis=0), axis=-1) / np.maximum(np.diff(gt[:, 0]), 1e-3)
            lim = QUALITY_RULES["max_speed_m_s"] if p["coord"] == "ground_plane_meters" else QUALITY_RULES["max_speed_bh_s"] * p["body_px"]
            if (d > lim).any():
                flags.add("possible_id_switch")
        p["evaluations"] = {}
        for h in p["horizons"]:
            reasons = sorted(flags & {"camera_cut", "possible_id_switch", "id_switch", "camera_motion", "low_tracker_confidence"})
            covered = gt[gt[:, 0] <= h + 1e-6] if len(gt) else gt
            if len(gt) == 0 or gt[-1, 0] < h - 0.15:
                reasons.append("recording_ended" if "recording_ended" in flags else ("track_lost" if "track_lost" in flags else "insufficient_ground_truth"))
            elif len(covered) < 2 or np.diff(np.r_[covered[:, 0], gt[gt[:, 0] > h][:1, 0]]).max(initial=0) > QUALITY_RULES["max_gt_gap_s"]:
                reasons.append("ground_truth_gap")
            if reasons:
                p["evaluations"][str(h)] = {"included": False, "exclusion_reasons": sorted(set(reasons))}
                continue
            sc = {}
            for m, mo in p["models"].items():
                sc[m] = score_forecast(mo, gt[:, 0], gt[:, 1:], h, p["coord"], p["body_px"])
            p["evaluations"][str(h)] = {"included": True, "scores": sc}
        out = {k: v for k, v in p.items() if k not in ("cam_frames", "future_frames")}
        out["flags"] = sorted(flags)
        out["actual_future"] = p["gt"]
        out.pop("gt")
        return out
