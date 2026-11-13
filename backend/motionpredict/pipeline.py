"""Per-session perception + forecasting pipeline with stage-level instrumentation.

Every stage appends a ``StageEvent`` with its wall-clock start, measured duration, summarised
inputs/outputs and a short plain-language explanation. These events are what the diagnostics
panel shows - they are measurements of the code that actually ran, not narrative.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np

from .calibration import GroundCalibration, estimate_height
from .camera_motion import CameraMotionEstimator
from .forecasting.constants import DT, MIN_OBS, NEIGHBOR_RADIUS
from .forecasting.service import VIRTUAL_REPLAY_CALIBRATION, Forecaster, UI_HORIZONS
from .kinematics import KinematicTrack
from .schemas import Detection, FrameResult, HeightEstimate, PipelineConfig, StageEvent, TrackOut
from .tracking import ByteTracker

BODY_HEIGHT_M = 1.7  # normalisation constant for image-mode model input only (never reported)
TIME_BACKWARDS_TOL_S = 0.05  # larger backward jumps (looped / seeked video, demo wrap) reset the session
TIME_BACKWARDS_WARNING = "Timestamps went backwards (video looped or seeked) - tracks were reset."


def now_ms() -> float:
    return time.time() * 1000.0


class StageRecorder:
    def __init__(self):
        self.events: list[StageEvent] = []

    @contextmanager
    def stage(self, stage: str, label: str, inputs: dict | None = None, explanation: str = ""):
        ev = StageEvent(stage=stage, label=label, t_start_ms=now_ms(), duration_ms=0.0, inputs=inputs or {}, explanation=explanation)
        t0 = time.perf_counter()
        try:
            yield ev
        except Exception as e:  # recorded, then re-raised to the caller
            ev.status = "error"
            ev.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            ev.duration_ms = (time.perf_counter() - t0) * 1000
            self.events.append(ev)

    def add(self, **kw) -> StageEvent:
        ev = StageEvent(t_start_ms=now_ms(), duration_ms=kw.pop("duration_ms", 0.0), **kw)
        self.events.append(ev)
        return ev


def band_points(points: np.ndarray, sigmas: list[list[float]], k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-track +-k sigma boundary of a path given per-point ellipse params."""
    pts = np.asarray(points, float)
    tang = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[1.0, 0.0]])
    left, right = [], []
    for p, t, (a, b, ang) in zip(pts, tang, sigmas):
        n = np.array([-t[1], t[0]])
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else np.array([0.0, 1.0])
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        C = R @ np.diag([a * a, b * b]) @ R.T
        s = math.sqrt(max(float(n @ C @ n), 0.0))
        left.append(p + k * s * n)
        right.append(p - k * s * n)
    return np.asarray(left), np.asarray(right)


class Session:
    def __init__(self, detector, forecaster: Forecaster, device: str = "cpu"):
        self.id = uuid.uuid4().hex[:12]
        # serialises process() (worker thread) against reset()/calibration changes (event loop)
        self.lock = threading.RLock()
        self.generation = 0  # bumped by every reset; results of frames started before a reset are stale
        self.detector = detector
        self.forecaster = forecaster
        self.device = device
        self.config = PipelineConfig()
        self.calibration: GroundCalibration | None = None
        self.tracker = ByteTracker()
        self.cam = CameraMotionEstimator()
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self, keep_ids: bool = False) -> None:
        """Clear every per-person buffer (called on stop / disconnect).

        ``keep_ids=True`` keeps the track-id counter so new tracks never reuse an earlier id
        (used after a timestamp discontinuity or a camera cut)."""
        with self.lock:
            self.generation += 1
            self.tracker.reset(keep_ids=keep_ids)
            self._reset_people()
            self.frames_processed = 0
            self.frames_received = 0
            self.frames_dropped = 0
            self.proc_times: deque[float] = deque(maxlen=30)  # end-to-end (receipt -> done)
            self.compute_times: deque[float] = deque(maxlen=30)  # worker start -> done
            self.queue_times: deque[float] = deque(maxlen=30)  # receipt -> worker start
            self.recv_times: deque[float] = deque(maxlen=31)  # wall-clock receipt times (input rate)
            self.last_coord = "image_pixels"
            self.last_wh = (0, 0)
            self._source: str | None = None

    def _reset_people(self) -> None:
        """Per-person / per-timeline state (tracks, kinematics, camera-motion reference, frame clock)."""
        self.cam.reset()
        self.kin: dict[int, dict[str, Any]] = {}
        self.last_frame_t: float | None = None
        self.frame_intervals: deque[float] = deque(maxlen=30)
        self.last_outs: list[dict] = []
        self._cam_now = None

    def note_received(self, received_ms: float, dropped_previous: bool = False) -> None:
        """Count a frame arriving at the server (before any queueing) for the input-rate statistic."""
        self.frames_received += 1
        if dropped_previous:
            self.frames_dropped += 1
        self.recv_times.append(received_ms)

    def _input_fps(self) -> float | None:
        r = list(self.recv_times)
        if len(r) >= 2 and r[-1] > r[0]:
            return (len(r) - 1) * 1000.0 / (r[-1] - r[0])
        iv = [x for x in self.frame_intervals if x > 0]  # fallback: frame timestamps (never negative, C4)
        return 1.0 / float(np.mean(iv)) if iv else None

    def apply_config(self, cfg: PipelineConfig) -> None:
        self.config = cfg
        self.tracker.high_thresh = cfg.det_conf
        self.tracker.new_track_thresh = min(cfg.det_conf + 0.1, 0.95)
        self.tracker.track_buffer_s = cfg.track_buffer_s

    # ------------------------------------------------------------ processing
    def process(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        t_s: float,
        received_ms: float,
        client_ts_ms: float | None,
        source: str,
        decode_ms: float,
        nbytes: int,
    ) -> FrameResult:
        with self.lock:
            return self._process(frame_bgr, frame_id, t_s, received_ms, client_ts_ms, source, decode_ms, nbytes)

    def _process(self, frame_bgr, frame_id, t_s, received_ms, client_ts_ms, source, decode_ms, nbytes) -> FrameResult:
        started = now_ms()  # worker start: queue wait = started - received, compute = done - started (C2)
        cfg = self.config
        rec = StageRecorder()
        H, W = frame_bgr.shape[:2]
        self.last_wh = (W, H)
        self._cam_now = None
        self._source = source
        warnings: list[str] = []
        if self.last_frame_t is not None and t_s < self.last_frame_t - TIME_BACKWARDS_TOL_S:
            # looped / seeked video or demo wrap: every per-person state refers to the old timeline (C4)
            self.tracker.reset(keep_ids=True)
            self._reset_people()
            warnings.append(TIME_BACKWARDS_WARNING)
        elif self.last_frame_t is not None and t_s < self.last_frame_t:
            t_s = self.last_frame_t  # small jitter backwards: hold the clock (no negative intervals)
        dt = 0.0 if self.last_frame_t is None else t_s - self.last_frame_t
        if self.last_frame_t is not None and dt > 0:
            self.frame_intervals.append(dt)
        self.last_frame_t = t_s
        cal = self.calibration
        st: dict[str, Any] = {"dets": [], "hi": [], "outs": [], "forecasts": {}, "cam": None}
        planned = ["detector", "pose", "tracker"] + (["camera_motion"] if cfg.camera_motion_check and source != "dataset_replay" else []) \
            + ["calibration", "kinematics", "forecast", "hypotheses", "probabilities"]

        rec.add(stage="frame_received", label="Frame received", duration_ms=decode_ms,
                inputs={"bytes": nbytes, "source": source},
                outputs={"width": W, "height": H, "frame_id": frame_id},
                explanation="Frame decoded from JPEG into an image array.")
        try:
            self._stages(rec, cfg, frame_bgr, t_s, dt, source, W, H, warnings, st)
        except Exception as e:  # a failing stage still yields a result: its event carries the error (status 'error')
            failed = next((ev for ev in rec.events if ev.status == "error"), None)
            if failed is None:
                failed = rec.add(stage="pipeline", label="Pipeline error", status="error", error=f"{type(e).__name__}: {e}",
                                 explanation="An error occurred outside a measured stage.")
            recorded = {ev.stage for ev in rec.events}
            for name in planned:
                if name not in recorded:
                    rec.add(stage=name, label=name.replace("_", " ").capitalize(), status="skipped",
                            outputs={"reason": f"upstream stage '{failed.stage}' failed"},
                            explanation="Not run for this frame because an earlier stage failed.")
            warnings.append(f"Stage '{failed.stage}' failed ({failed.error}); later stages were skipped for this frame.")

        # 6. assemble (tracks are reported even if a later stage such as forecasting failed)
        outs, forecasts = st["outs"], st["forecasts"]
        tracks_out = []
        try:
            for o in outs:
                tr = o["tr"]
                f = forecasts.get(tr.track_id)
                if f is not None:
                    self._attach_image_geometry(f, W, H)
                tracks_out.append(TrackOut(
                    track_id=tr.track_id, state=tr.state, age_s=round(t_s - tr.start_t, 3), hits=tr.hits,
                    time_since_update_s=round(t_s - tr.last_t, 3),
                    tracker_confidence=round(tr.confidence(t_s, self.tracker.track_buffer_s), 3),
                    bbox=[round(v, 1) for v in tr.kf.box.tolist()],
                    det_confidence=round(tr.det_conf, 3), footpoint_px=o["foot"],
                    keypoints=tr.current_keypoints(t_s) if (tr.matched_this_frame and cfg.pose) else None,
                    history_px=o["hist_px"], history_world=o["hist_w"],
                    kinematics=o["k_px"], kinematics_world=o["k_w"], height=o["height"], forecast=f,
                ))
        except Exception as e:
            tracks_out = []
            rec.add(stage="assemble", label="Result assembly", status="error", error=f"{type(e).__name__}: {e}",
                    explanation="Building the per-track output failed; no tracks are reported for this frame.")
            warnings.append(f"Stage 'assemble' failed ({type(e).__name__}: {e}); no tracks reported for this frame.")
        for rid in self.tracker.removed_ids:
            self.kin.pop(rid, None)

        self.frames_processed += 1
        done = now_ms()
        proc = done - received_ms
        self.proc_times.append(proc)
        self.compute_times.append(done - started)
        self.queue_times.append(max(started - received_ms, 0.0))
        fps_in = self._input_fps()
        compute_mean = float(np.mean(self.compute_times))
        cam = st["cam"]
        return FrameResult(
            session_id=self.id, frame_id=frame_id, source=source, client_ts_ms=client_ts_ms,
            server_received_ms=received_ms, server_done_ms=done, frame_width=W, frame_height=H,
            processing_ms=round(proc, 2), tracks=tracks_out, detections=len(st["hi"]), events=rec.events,
            warnings=warnings, camera_calibration=cal.to_dict() if cal else {"status": "uncalibrated", "coordinate_system": "image_pixels"},
            camera_motion=cam,
            stats={"frames_received": self.frames_received, "frames_processed": self.frames_processed,
                   "frames_dropped": self.frames_dropped, "input_fps": None if fps_in is None else round(float(fps_in), 2),
                   "inference_fps": round(1000.0 / compute_mean, 2) if compute_mean > 0 else None,
                   "compute_ms_mean": round(compute_mean, 2),
                   "queue_ms_mean": round(float(np.mean(self.queue_times)), 2),
                   "mean_processing_ms": round(float(np.mean(self.proc_times)), 2), "device": self.device,
                   "forecaster_active": next(iter(forecasts.values())).prediction_source if forecasts else None},
        )

    def _stages(self, rec: StageRecorder, cfg: PipelineConfig, frame_bgr: np.ndarray, t_s: float, dt: float, source: str,
                W: int, H: int, warnings: list[str], st: dict) -> None:
        """Stages 1-5; results are written into ``st`` as they become available (so a later failure keeps them)."""
        # 1. detection (+ pose, fused)
        with rec.stage("detector", "Person detector executed",
                       {"model": self.detector.weights, "conf_low": 0.1, "conf_high": cfg.det_conf, "iou": cfg.det_iou, "imgsz": cfg.imgsz, "device": self.device},
                       "YOLO looks for the 'person' class and returns boxes with confidence scores.") as ev:
            dets: list[Detection] = self.detector(frame_bgr, conf=0.1, iou=cfg.det_iou, imgsz=cfg.imgsz)
            hi = [d for d in dets if d.confidence >= cfg.det_conf]
            st["dets"], st["hi"] = dets, hi
            ev.outputs = {"people_detected": len(hi), "low_score_candidates": len(dets) - len(hi),
                          "confidences": [round(d.confidence, 3) for d in hi][:10]}
            ev.confidence = float(np.mean([d.confidence for d in hi])) if hi else None
        pose_on = cfg.pose and getattr(self.detector, "supports_pose", False)
        run_pose = pose_on and self.frames_processed % cfg.pose_every == 0
        with rec.stage("pose", "Pose estimation executed", {"model": getattr(self.detector, "pose_weights", None), "people": len(hi),
                                                            "every_nth_frame": cfg.pose_every},
                       "Estimates 17 body joints per detected person (separate YOLO-pose pass, matched to boxes). On frames in between, "
                       "the last joints are carried along with the person's box for up to 0.6 s. Used for the skeleton overlay and the "
                       "upright/full-body check for height. No face analysis.") as ev:
            if run_pose:
                n_kp = self.detector.pose(frame_bgr, hi, imgsz=cfg.imgsz)
                ev.outputs = {"people_with_keypoints": n_kp, "keypoints_per_person": 17}
            else:
                ev.status = "skipped"
                ev.outputs = {"reason": ("throttled: pose runs every %d frames" % cfg.pose_every) if pose_on
                              else ("disabled" if not cfg.pose else "detector has no pose model")}

        # 2. tracking
        with rec.stage("tracker", "Tracker updated (ByteTrack)", {"detections": len(dets), "dt_s": round(dt, 4)},
                       "Detections are linked to existing tracks by box overlap; low-confidence boxes can only extend existing tracks.") as ev:
            stats = self.tracker.update(dets, t_s)
            ev.outputs = stats

        # 3. camera motion
        if cfg.camera_motion_check and source != "dataset_replay":
            with rec.stage("camera_motion", "Camera motion check",
                           {"method": "LK optical flow on background",
                            "threshold_frac_width_per_s": self.cam.threshold_frac_per_s},
                           "Measures how fast the background moved (per second), to tell camera motion apart from people moving.") as ev:
                cam = self.cam.update(frame_bgr, [d.bbox for d in hi], dt)
                self._cam_now = cam
                st["cam"] = cam
                ev.outputs = cam
            if cam["camera_moving"]:
                warnings.append("Camera appears to be moving - apparent pedestrian velocities include camera motion.")

        # 4. calibration / ground mapping + kinematics
        cal = self.calibration
        coord = "ground_plane_meters" if cal else "image_pixels"
        rec.add(stage="calibration", label="Camera calibration status",
                outputs={"status": cal.status if cal else "uncalibrated", "coordinate_system": coord,
                         "height_estimation": bool(cal and cal.camera_height_m)},
                explanation="Without calibration, positions and speeds are in image pixels and metric height is unknown.")
        if not cal:
            warnings.append("Uncalibrated camera: speeds are in px/s and heights are unknown.")

        self.last_coord = coord
        with rec.stage("kinematics", "Kinematic state updated", {"filter": "constant-acceleration Kalman", "coordinate_system": coord},
                       "Each person's ground-contact point is smoothed and differentiated to get speed, heading, acceleration and turn rate.") as ev:
            outs = self._update_kinematics(t_s, W, H)
            self.last_outs = outs
            st["outs"] = outs
            ev.outputs = {"tracks": len(outs),
                          "speeds": {str(o["id"]): round(o["k"].speed, 3) for o in outs if o["k"] is not None},
                          "units": "m/s" if cal else "px/s"}

        # 5. forecasting
        with rec.stage("forecast", "Forecast model executed",
                       {"requested": cfg.forecaster, "horizon_s": cfg.horizon_s, "hypotheses": cfg.num_hypotheses},
                       "Predicts where each confirmed person may be over the next seconds.") as ev:
            forecasts, finfo = self._forecast(outs, coord)
            ev.outputs = finfo
            if finfo.get("fallback_reason"):
                warnings.append(finfo["fallback_reason"])
        st["forecasts"] = forecasts
        n_h = sum(len(f.trajectories) for f in forecasts.values())
        rec.add(stage="hypotheses", label="Trajectory hypotheses generated",
                outputs={"tracks_forecast": len(forecasts), "hypotheses": n_h,
                         "per_track": {str(k): [t.id for t in f.trajectories] for k, f in forecasts.items()}},
                explanation="Each hypothesis is one possible future path with growing uncertainty.")
        prob_ev = {}
        for k, f in forecasts.items():
            p = [t.probability for t in f.trajectories]
            prob_ev[str(k)] = {"type": f.probability_type, "values": [None if x is None else round(x, 3) for x in p],
                               "residual": f.residual_probability, "calibration_status": f.calibration_status}
        any_learned = any(f.prediction_source == "trained_model" for f in forecasts.values())
        rec.add(stage="probabilities", label="Probabilities calculated", status="ok" if any_learned else "skipped",
                outputs=prob_ev,
                explanation=("Softmax over the model's mode scores, temperature-scaled on validation data. Labelled "
                             "'calibrated probability' only if the checkpoint passed both held-out ECE gates (all modes and top mode), "
                             "coordinates are metric and the camera calibration is verified (or dataset replay); otherwise "
                             "'uncalibrated weight'. The reason is in calibration_status.")
                if any_learned else "Physics baseline produces uncertainty bands only - no route probabilities.")

    # ----------------------------------------------------------- internals
    def _update_kinematics(self, t: float, W: int, H: int) -> list[dict]:
        cfg, cal = self.config, self.calibration
        outs = []
        for tr in self.tracker.tracks:
            if tr.state == "tentative":
                continue
            st = self.kin.get(tr.track_id)
            box = tr.last_det_box if tr.last_det_box is not None else tr.kf.box
            bh = max(float(box[3] - box[1]), 4.0)
            if st is None:
                st = {"px": KinematicTrack("px", q=1.0, r=1.0, history_s=cfg.history_s), "w": None,
                      "hpx": deque(maxlen=60), "heights": deque(maxlen=30)}
                self.kin[tr.track_id] = st
            st["hpx"].append(bh)
            med_h = float(np.median(st["hpx"]))
            kp: KinematicTrack = st["px"]
            kp.history_s = cfg.history_s
            kp.scale = med_h / BODY_HEIGHT_M  # px per pseudo-meter (normalisation only)
            foot = np.array([(box[0] + box[2]) / 2, box[3]])
            if tr.matched_this_frame:
                kp.update(t, foot, r=(0.02 * med_h) ** 2, q=4.0 * kp.scale**2)
            else:
                kp.coast(t)
            kw = None
            if cal is not None:
                if st["w"] is None:
                    st["w"] = KinematicTrack("m", q=4.0, r=0.05**2, history_s=cfg.history_s)
                kw = st["w"]
                kw.history_s = cfg.history_s
                g = cal.to_ground(np.array([[foot[0] / W, foot[1] / H]]))[0]
                if tr.matched_this_frame and np.isfinite(g).all():
                    # measurement noise grows with distance (pixel error maps to more meters far away)
                    g2 = cal.to_ground(np.array([[foot[0] / W, (foot[1] + 0.02 * med_h) / H]]))[0]
                    r = max(float(np.sum((g2 - g) ** 2)), 0.03**2)
                    kw.update(t, g, r=r)
                else:
                    kw.coast(t)
            else:
                st["w"] = None
            # height
            hest = estimate_height(cal, box.tolist(), tr.current_keypoints(t) if tr.matched_this_frame else None, W, H)
            if hest.status == "calibrated":
                st["heights"].append((hest.meters, hest.interval_m))
            if st["heights"] and cal is not None and cal.camera_height_m:
                ms = np.array([h for h, _ in st["heights"]])
                lo = np.median([i[0] for _, i in st["heights"]])
                hi_ = np.median([i[1] for _, i in st["heights"]])
                m = float(np.median(ms))
                hest = HeightEstimate(status="calibrated", meters=round(m, 3),
                                      interval_m=[round(float(lo), 3), round(float(hi_), 3)],
                                      relative_image_height=bh / H,
                                      reason=f"median of {len(ms)} per-frame horizon-ratio estimates (approximate)"
                                             + ("" if hest.status == "calibrated" else f"; current frame: {hest.reason}"))
            hist_px = [[round(x, 1), round(y, 1), round(tt - t, 3)] for tt, x, y in kp.history]
            hist_w = [[round(x, 3), round(y, 3), round(tt - t, 3)] for tt, x, y in kw.history] if kw else None
            outs.append({"id": tr.track_id, "tr": tr, "st": st, "foot": [round(float(foot[0]), 1), round(float(foot[1]), 1)],
                         "k_px": kp.state(), "k_w": kw.state() if kw else None,
                         "k": (kw.state() if kw else kp.state()) or kp.state(), "height": hest,
                         "hist_px": hist_px, "hist_w": hist_w, "kin": kw if kw else kp})
        return outs

    def forecast_now(self, kind: str, horizon: float, k: int, track_ids: set[int] | None = None) -> tuple[dict, dict]:
        """Forecast from the *current* state only (used by the evaluation engine)."""
        with self.lock:
            return self._forecast_now(kind, horizon, k, track_ids)

    def _forecast_now(self, kind: str, horizon: float, k: int, track_ids: set[int] | None) -> tuple[dict, dict]:
        saved = self.config
        self.config = saved.model_copy(update={"forecaster": kind, "horizon_s": horizon, "num_hypotheses": k})
        try:
            outs = [o for o in self.last_outs if track_ids is None or o["id"] in track_ids]
            res, info = self._forecast(outs, self.last_coord)
            for f in res.values():
                self._attach_image_geometry(f, self.last_wh[0], self.last_wh[1])
            return res, info
        finally:
            self.config = saved

    def _forecast(self, outs: list[dict], coord: str):
        cfg = self.config
        horizon = cfg.horizon_s
        use_learned = cfg.forecaster == "learned" and self.forecaster.available
        info: dict[str, Any] = {"source": "trained_model" if use_learned else "physics_baseline"}
        if cfg.forecaster == "learned" and not self.forecaster.available:
            info["fallback_reason"] = f"Learned model unavailable ({self.forecaster.load_error}); using physics baseline."
        eligible = [o for o in outs if o["tr"].state == "confirmed" and len(o["kin"].history) >= 3 and o["k"] is not None]
        cal = self.calibration
        cam_cal = VIRTUAL_REPLAY_CALIBRATION if (cal is not None and self._source == "dataset_replay") else (cal.status if cal else None)
        res = {}
        t0 = time.perf_counter()
        if use_learned:
            items = []
            for o in eligible:
                kin: KinematicTrack = o["kin"]
                obs, nv = kin.resampled()
                if nv < MIN_OBS:
                    continue
                scale = 1.0 if coord == "ground_plane_meters" else 1.0 / kin.scale
                cand = []
                for o2 in eligible:
                    if o2 is o:
                        continue
                    k2: KinematicTrack = o2["kin"]
                    p2 = k2.kf.x[:2]
                    d = float(np.hypot(*(p2 - obs[-1]))) * scale
                    if d <= NEIGHBOR_RADIUS:
                        cand.append((d, [*p2, *k2.velocity_per_step()]))
                cand.sort(key=lambda c: c[0])  # nearest first, as in training (the model keeps the closest MAX_NEIGHBORS)
                nbrs = [c[1] for c in cand]
                items.append({"track_id": o["id"], "obs": obs, "n_valid": nv, "nbrs": nbrs, "scale": scale})
            for f in self.forecaster.learned_batch(items, horizon, coord, cfg.num_hypotheses, camera_calibration=cam_cal):
                res[f.track_id] = f
            info["model_version"] = self.forecaster.meta.get("version")
            info["model_input"] = {"steps": 8, "dt_s": DT, "tracks": len(items),
                                   "normalisation": "none (meters)" if coord == "ground_plane_meters" else "body-height (1.7/median bbox px)"}
        else:
            for o in eligible:
                kin: KinematicTrack = o["kin"]
                _, nv = kin.resampled()
                res[o["id"]] = self.forecaster.baseline(o["id"], kin.kf, horizon, coord, [], nv,
                                                        unit_scale=1.0 if coord == "ground_plane_meters" else kin.scale,
                                                        num_hypotheses=cfg.num_hypotheses)
        info["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        info["tracks_forecast"] = len(res)
        info["horizons_s"] = [h for h in UI_HORIZONS if h <= horizon + 1e-6]
        return res, info

    def _attach_image_geometry(self, f, W: int, H: int) -> None:
        cal = self.calibration
        for tr in f.trajectories:
            pts = np.asarray(tr.points)
            left, right = band_points(pts, tr.sigma)
            if cal is not None:
                to_img = lambda p: (cal.to_image(p) * np.array([W, H]))  # noqa: E731
                tr.image_points = np.round(to_img(pts), 1).tolist()
                tr.band_left = np.round(to_img(left), 1).tolist()
                tr.band_right = np.round(to_img(right), 1).tolist()
            else:
                tr.image_points = np.round(pts, 1).tolist()
                tr.band_left = np.round(left, 1).tolist()
                tr.band_right = np.round(right, 1).tolist()
