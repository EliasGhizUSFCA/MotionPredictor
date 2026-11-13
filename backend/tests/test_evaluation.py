"""Prediction-vs-reality engine: causality, scoring and reporting."""

import json

import cv2
import numpy as np
import pytest
from conftest import HAS_DATA, RAW

from motionpredict.evaluation.engine import EvaluationRun
from motionpredict.evaluation.scoring import score_forecast
from motionpredict.forecasting.service import Forecaster
from motionpredict.schemas import Detection


class BlobDetector:
    """Finds bright rectangles: a deterministic stand-in for YOLO on synthetic video."""

    weights = "blob-test-detector"
    is_pose = False

    def __call__(self, frame, **_):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n, _, stats, _ = cv2.connectedComponentsWithStats((g > 128).astype(np.uint8))
        return [Detection(bbox=[float(x), float(y), float(x + w), float(y + h)], confidence=0.9)
                for x, y, w, h, a in stats[1:] if a > 50]


def make_video(path, diverge_at_frame: int, turn: str, n: int = 120, fps: int = 10):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    x, y = 40.0, 120.0
    for i in range(n):
        if i < diverge_at_frame:
            x += 2.0
        elif turn == "straight":
            x += 2.0
        else:
            y += 2.0
        f = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(f, (int(x) - 6, int(y) - 30), (int(x) + 6, int(y)), (255, 255, 255), -1)
        wr.write(f)
    wr.release()


def run_video(tmp_path, name, turn):
    vid = tmp_path / f"{name}.avi"
    make_video(vid, 60, turn)
    run = EvaluationRun(tmp_path / "evals", run_id=name)
    run.run({"interval_s": 0.5, "horizons": [1.0, 2.0], "process_fps": 10, "pipeline": {"camera_motion_check": False}},
            BlobDetector(), Forecaster(checkpoint=None), video_path=vid, keep_video=True)
    assert run.status == "done", run.error
    return [json.loads(line) for line in (run.dir / "predictions.jsonl").read_text().splitlines()], run.dir


def test_forecasts_never_use_future_frames(tmp_path):
    a, _ = run_video(tmp_path, "straight", "straight")
    b, _ = run_video(tmp_path, "turn", "turn")
    t_div = 6.0
    before_a = {p["t"]: p for p in a if p["t"] < t_div - 1e-6}
    before_b = {p["t"]: p for p in b if p["t"] < t_div - 1e-6}
    assert before_a and before_a.keys() == before_b.keys()
    for t in before_a:
        strip = lambda ms: {m: {k: v for k, v in mo.items() if k != "latency_ms"} for m, mo in ms.items()}  # noqa: E731
        assert strip(before_a[t]["models"]) == strip(before_b[t]["models"]), f"forecast at t={t} depends on frames after it"
        assert before_a[t]["causality"]["latest_frame_t_used"] == t
    # ...but the evaluation of forecasts spanning the divergence differs (future is used only for scoring)
    t_probe = max(t for t in before_a if t + 2.0 > t_div)
    sa = before_a[t_probe]["evaluations"]["2.0"]
    sb = [p for p in b if p["t"] == t_probe][0]["evaluations"]["2.0"]
    assert sa["included"] and sb["included"]
    assert sb["scores"]["baseline"]["fde_top1"] > sa["scores"]["baseline"]["fde_top1"] + 5


def test_summary_and_exports(tmp_path):
    _, d = run_video(tmp_path, "s", "straight")
    s = json.loads((d / "summary.json").read_text())
    assert s["coordinate_system"] == "image_pixels" and s["unit"] == "px"
    d1 = s["per_model_horizon"]["baseline@1.0"]
    assert d1["n"] > 3 and d1["ade_top1"]["mean"] < 3.0  # straight constant-velocity walker: tiny error
    assert (d / "predictions.csv").read_text().startswith("prediction_id,")
    assert "prediction vs. reality" in (d / "report.html").read_text()


def test_score_definitions():
    mo = {"prediction_source": "trained_model", "trajectories": [
        {"id": "A", "probability": 0.7, "times": [0, 1, 2], "points": [[0, 0], [1, 0], [2, 0]], "sigma": [[0.1, 0.1, 0]] * 3},
        {"id": "B", "probability": 0.3, "times": [0, 1, 2], "points": [[0, 0], [0, 1], [0, 2]], "sigma": [[0.1, 0.1, 0]] * 3}]}
    gt_t, gt = np.array([0, 1, 2.0]), np.array([[0, 0], [0, 1], [0, 2.0]])  # person went the low-probability way
    s = score_forecast(mo, gt_t, gt, 2.0, "ground_plane_meters", 1.0)
    assert s["top_id"] == "A" and s["best_id"] == "B"
    assert s["min_fde"] == pytest.approx(0) and s["fde_top1"] == pytest.approx(np.hypot(2, 2))
    assert s["min_fde"] < s["fde_top1"], "best-of-K must be reported separately from the top-1 path"
    assert s["w_ade"] == pytest.approx(0.7 * s["ade_top1"])
    assert s["miss_top1"] and not s["miss_k"]


@pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded")
def test_dataset_scene_run_uses_annotations(tmp_path):
    run = EvaluationRun(tmp_path)
    run.run({"interval_s": 2.0, "max_duration_s": 30, "horizons": [1.0, 3.0]}, None, Forecaster(), scene="zara1", raw_dir=RAW)
    assert run.status == "done", run.error
    s = json.loads((run.dir / "summary.json").read_text())
    assert s["coordinate_system"] == "ground_plane_meters" and "human annotations" in s["ground_truth"]
    assert s["per_model_horizon"]["learned@1.0"]["n"] > 0
