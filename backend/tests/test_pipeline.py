import json
import time

import numpy as np
import pytest
from conftest import HAS_DATA, RAW

from motionpredict.forecasting.service import Forecaster
from motionpredict.pipeline import Session
from motionpredict.schemas import Detection, FrameResult

EXPECTED_STAGES = ["frame_received", "detector", "pose", "tracker", "calibration", "kinematics", "forecast", "hypotheses", "probabilities"]


class ScriptedDetector:
    weights = "scripted-test-detector"
    is_pose = False

    def __init__(self):
        self.dets = []

    def __call__(self, frame, **_):
        return self.dets


def run_walker(forecaster, kind="learned", n=30):
    det = ScriptedDetector()
    s = Session(det, forecaster)
    s.apply_config(s.config.model_copy(update={"forecaster": kind, "camera_motion_check": False}))
    frame = np.zeros((480, 640, 3), np.uint8)
    res = None
    for i in range(n):
        x = 100 + 6 * i
        det.dets = [Detection(bbox=[x - 20, 150, x + 20, 350], confidence=0.9)]
        res = s.process(frame, i, i * 0.1, time.time() * 1000, None, "browser", 0.1, 1000)
    return s, res


def test_baseline_has_no_probabilities_and_labels_units():
    fc = Forecaster(checkpoint=None)
    assert not fc.available
    s, res = run_walker(fc, "learned")  # learned requested but unavailable -> labelled fallback
    assert [e.stage for e in res.events] == EXPECTED_STAGES
    t = res.tracks[0]
    assert t.kinematics.units == "px" and t.kinematics.speed == pytest.approx(60, rel=0.15)
    assert t.height.status == "uncalibrated" and t.height.meters is None
    f = t.forecast
    assert f.prediction_source == "physics_baseline" and f.probability_type == "none"
    assert all(tr.probability is None for tr in f.trajectories)
    assert any("unavailable" in w for w in res.warnings)
    # forecast moves forward in +x
    assert f.trajectories[0].points[-1][0] > t.footpoint_px[0]
    FrameResult.model_validate_json(res.model_dump_json())
    s.reset()
    assert s.kin == {} and s.tracker.tracks == []


@pytest.mark.skipif(not (RAW.parent.parent / "models/socialmdn_zara1/model.pt").exists(), reason="no trained checkpoint")
def test_learned_in_image_space_is_uncalibrated_weight():
    fc = Forecaster()
    _, res = run_walker(fc, "learned")
    f = res.tracks[0].forecast
    assert f.prediction_source == "trained_model"
    assert f.probability_type == "uncalibrated_weight"
    ps = [tr.probability for tr in f.trajectories]
    assert abs(sum(ps) + f.residual_probability - 1) < 1e-4
    assert ps == sorted(ps, reverse=True)


@pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded")
def test_dataset_replay_metric_mode():
    from motionpredict.demo import ReplayDetector, SceneReplay

    rp = SceneReplay(RAW, "zara1")
    det = ReplayDetector()
    s = Session(det, Forecaster())
    s.calibration = rp.calibration
    s.apply_config(s.config.model_copy(update={"pose": False, "camera_motion_check": False}))
    for i in range(40):
        fr, t, _ = rp.step(det)
        res = s.process(fr, i, t, time.time() * 1000, None, "dataset_replay", 0, 0)
    tr = [x for x in res.tracks if x.forecast]
    assert tr, "expected forecasts after 4 s of replay"
    k = tr[0].kinematics_world
    assert k.units == "m" and 0 <= k.speed < 4
    assert tr[0].forecast.coordinate_system == "ground_plane_meters"
    json.loads(res.model_dump_json())


class FailingDetector(ScriptedDetector):
    def __call__(self, frame, **_):
        raise RuntimeError("detector exploded")


def test_failing_stage_still_returns_result_with_error_event():
    s = Session(FailingDetector(), Forecaster(checkpoint=None))
    s.apply_config(s.config.model_copy(update={"camera_motion_check": False}))
    res = s.process(np.zeros((120, 160, 3), np.uint8), 1, 0.0, time.time() * 1000, None, "browser", 0.1, 10)
    ev = {e.stage: e for e in res.events}
    assert ev["detector"].status == "error" and "detector exploded" in ev["detector"].error
    assert [e.stage for e in res.events] == EXPECTED_STAGES
    assert all(ev[st].status == "skipped" for st in EXPECTED_STAGES[2:])
    assert res.tracks == [] and any("detector" in w for w in res.warnings)
    FrameResult.model_validate_json(res.model_dump_json())


def test_timestamps_backwards_reset_tracks_with_new_ids():
    from motionpredict.pipeline import TIME_BACKWARDS_WARNING

    s, res = run_walker(Forecaster(checkpoint=None), "baseline", n=30)
    old_ids = {t.track_id for t in res.tracks}
    frame = np.zeros((480, 640, 3), np.uint8)
    s.detector.dets = [Detection(bbox=[80, 150, 120, 350], confidence=0.9)]
    res = s.process(frame, 31, 0.0, time.time() * 1000, None, "browser", 0.1, 1000)  # video looped
    assert TIME_BACKWARDS_WARNING in res.warnings
    assert all(t.track_id not in old_ids for t in res.tracks)  # ids are never reused
    assert all(t.time_since_update_s >= 0 for t in res.tracks)
    for i in range(1, 15):  # keeps working after the wrap (no IndexError from out-of-order history)
        s.detector.dets = [Detection(bbox=[80 + 6 * i, 150, 120 + 6 * i, 350], confidence=0.9)]
        res = s.process(frame, 31 + i, i * 0.1, time.time() * 1000, None, "browser", 0.1, 1000)
    assert TIME_BACKWARDS_WARNING not in res.warnings and res.tracks[0].forecast is not None
    assert res.stats["input_fps"] is None or res.stats["input_fps"] > 0
    assert min(s.frame_intervals) > 0


def test_stats_split_queue_and_compute_time():
    det = ScriptedDetector()
    s = Session(det, Forecaster(checkpoint=None))
    frame = np.zeros((120, 160, 3), np.uint8)
    for i in range(3):
        recv = time.time() * 1000 - 200  # frame waited ~200 ms before the worker started
        s.note_received(recv)
        res = s.process(frame, i, i * 0.1, recv, None, "browser", 0.1, 10)
    st = res.stats
    assert st["queue_ms_mean"] >= 190 and st["compute_ms_mean"] < st["mean_processing_ms"]
    assert st["inference_fps"] == pytest.approx(1000 / st["compute_ms_mean"], rel=0.05)
    assert res.processing_ms >= 190  # end-to-end stays receipt -> done


def test_baseline_honours_num_hypotheses():
    fc = Forecaster(checkpoint=None)
    for k, ids in [(1, ["CV"]), (2, ["CV", "CA"]), (5, ["CV", "CA"])]:
        det = ScriptedDetector()
        s = Session(det, fc)
        s.apply_config(s.config.model_copy(update={"forecaster": "baseline", "num_hypotheses": k, "camera_motion_check": False}))
        for i in range(10):
            det.dets = [Detection(bbox=[100 + 6 * i, 150, 140 + 6 * i, 350], confidence=0.9)]
            res = s.process(np.zeros((480, 640, 3), np.uint8), i, i * 0.1, time.time() * 1000, None, "browser", 0.1, 10)
        f = res.tracks[0].forecast
        assert [t.id for t in f.trajectories] == ids
        assert any("at most 2 motion assumptions" in w for w in f.warnings) == (k != 2)


def test_dense_times_end_at_exact_horizon():
    from motionpredict.forecasting.service import dense_times

    for h in (0.5, 1.0, 2.5, 3.0, 4.5, 5.0):
        t = dense_times(h)
        assert t[0] == 0 and t[-1] == pytest.approx(h) and np.all(np.diff(t) > 0)


def test_probability_gate_needs_both_eces_metric_coords_and_verified_camera():
    fc = Forecaster(checkpoint=None)
    fc.model = object()  # pretend loaded; only the labelling logic is tested
    fc.meta = {"temperature": 1.0, "held_out_test_scene": "zara1",
               "calibration": {"test_ece_all_modes": 0.03, "test_ece_top_mode": 0.19}}
    pt, why = fc.probability_label("ground_plane_meters", "calibrated_verified")
    assert pt == "uncalibrated_weight" and "top-mode ECE 0.190" in why and not fc.validated
    fc.meta["calibration"]["test_ece_top_mode"] = 0.06
    assert fc.validated
    assert fc.probability_label("ground_plane_meters", "calibrated_verified")[0] == "calibrated_probability"
    assert fc.probability_label("ground_plane_meters", "virtual_dataset_replay")[0] == "calibrated_probability"
    for cam in ("calibrated_poor_fit", "calibrated_unverified", None):
        pt, why = fc.probability_label("ground_plane_meters", cam)
        assert pt == "uncalibrated_weight" and "calibrated_verified" in why
    assert fc.probability_label("image_pixels", "calibrated_verified")[0] == "uncalibrated_weight"


@pytest.mark.skipif(not (RAW.parent.parent / "models/socialmdn_zara1/model.pt").exists(), reason="no trained checkpoint")
def test_active_checkpoint_top_mode_ece_fails_gate():
    fc = Forecaster()
    ece_top = fc.meta["calibration"]["test_ece_top_mode"]
    if ece_top >= 0.10:
        assert not fc.validated
        assert fc.probability_label("ground_plane_meters", "virtual_dataset_replay")[0] == "uncalibrated_weight"


def test_neighbours_passed_nearest_first():
    fc = Forecaster(checkpoint=None)
    seen = []
    fc.model = object()
    fc.learned_batch = lambda items, *a, **k: seen.extend(items) or []  # type: ignore[method-assign]
    det = ScriptedDetector()
    s = Session(det, fc)
    s.apply_config(s.config.model_copy(update={"forecaster": "learned", "camera_motion_check": False}))
    xs = [300, 560, 330, 420]  # ego at 300; neighbours in non-distance order
    for i in range(30):
        seen.clear()
        det.dets = [Detection(bbox=[x - 10 + i, 150, x + 10 + i, 350], confidence=0.9) for x in xs]
        s.process(np.zeros((480, 900, 3), np.uint8), i, i * 0.1, time.time() * 1000, None, "browser", 0.1, 10)
    ego = next(it for it in seen if len(it["nbrs"]) >= 2)
    d = [np.hypot(n[0] - ego["obs"][-1][0], n[1] - ego["obs"][-1][1]) for n in ego["nbrs"]]
    assert d == sorted(d)


def test_camera_motion_threshold_is_per_second():
    from motionpredict.camera_motion import CameraMotionEstimator

    rng = np.random.default_rng(0)
    base = (rng.random((240, 1200)) * 255).astype(np.uint8)
    base = np.repeat(base[:, :, None], 3, 2)

    def moving(fps: float, rate_frac_per_s: float) -> bool:
        est = CameraMotionEstimator()
        shift = rate_frac_per_s * 320 / fps  # pixels per frame (frame width 320)
        out = None
        for i in range(3):
            x0 = int(round(100 + i * shift))
            out = est.update(np.ascontiguousarray(base[:, x0 : x0 + 320]), [], 1.0 / fps)
        return out["camera_moving"]

    for fps in (5, 10, 20):
        assert moving(fps, 0.15) and not moving(fps, 0.0)
