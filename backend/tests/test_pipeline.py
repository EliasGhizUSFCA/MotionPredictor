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
