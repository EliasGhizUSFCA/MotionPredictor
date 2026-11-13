import pytest
from conftest import HAS_DATA
from fastapi.testclient import TestClient

from motionpredict.server import app
from test_calibration import GROUND, norm, project


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["device"]["device"] in ("cpu", "cuda", "mps")
    assert h["privacy"]["face_recognition"] is False


def test_calibration_fit_endpoint(client):
    img = [norm(project(x, y, 0)) for x, y in GROUND]
    r = client.post("/api/calibration/fit", json={"image_points": img, "world_points": [list(g) for g in GROUND], "camera_height_m": 3})
    assert r.status_code == 200 and r.json()["status"] == "calibrated_verified"
    r = client.post("/api/calibration/fit", json={"image_points": img[:3], "world_points": [list(g) for g in GROUND[:3]]})
    assert r.status_code == 400


def test_explain_rule_based(client):
    result = {"tracks": [{"track_id": 3, "state": "confirmed", "age_s": 2.0, "tracker_confidence": 0.8,
                          "height": {"status": "uncalibrated", "reason": "camera not calibrated"},
                          "kinematics": {"units": "px", "speed": 50.0, "heading_deg": 0.0, "turn_rate_deg_s": 0.0,
                                         "movement_state": "moving", "speed_category": None},
                          "forecast": None}], "warnings": []}
    r = client.post("/api/explain", json={"track_id": 3, "use_llm": False, "result": result}).json()
    assert r["ok"] and "Track 3" in r["rule_based"] and "px/s" in r["rule_based"] and "unknown" in r["rule_based"]


@pytest.mark.skipif(not HAS_DATA, reason="dataset not downloaded")
def test_demo_websocket(client):
    with client.websocket_connect("/ws/demo?scene=zara1&speed=4") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["demo"]["scene"] == "zara1"
        msg = ws.receive_json()
        assert msg["type"] == "result" and msg["source"] == "dataset_replay"
        assert msg["camera_calibration"]["coordinate_system"] == "ground_plane_meters"


def test_internal_dirs_are_not_runs(client, tmp_path, monkeypatch):
    import motionpredict.server as srv

    monkeypatch.setattr(srv, "EVAL_DIR", tmp_path)
    (tmp_path / "_uploads").mkdir()
    (tmp_path / "_uploads" / "kept.mp4").write_bytes(b"x")
    for rid in ("_uploads", "..", ".hidden"):
        assert client.delete(f"/api/eval/runs/{rid}").status_code in (404, 405)
        assert client.get(f"/api/eval/runs/{rid}").status_code in (404, 405)
    assert (tmp_path / "_uploads" / "kept.mp4").exists()


def _fake_run(root, rid, coord, model_version, errs, sha="abc"):
    import json

    d = root / rid
    d.mkdir()
    sfx = "" if coord == "ground_plane_meters" else "_bh"
    meta = {"source": {"type": "video", "scene": None, "sha256": sha}, "tags": {}, "coordinate_system": coord,
            "models": {"learned": {"version": model_version}}, "eval_config": {"interval_s": 1.0, "models": ["learned"]},
            "pipeline_config": {"det_conf": 0.35}, "calibration": None}
    (d / "run.json").write_text(json.dumps(meta))
    st = lambda v: {"mean": v, "median": v, "p90": v, "n": 3}  # noqa: E731
    pmh = {"learned@1.0": {"n": 3, "ade_top1": st(10.0), "fde_top1": st(20.0), "min_ade": st(8.0), "min_fde": st(15.0),
                           "ade_top1_bh": st(0.1), "fde_top1_bh": st(0.2), "min_ade_bh": st(0.08), "min_fde_bh": st(0.15),
                           "miss_rate_top1": 0.0, "coverage90": 0.9, "coverage50": 0.5, "latency_ms": 3.0,
                           "reliability": {"available": True, "ece": 0.1}}}
    (d / "summary.json").write_text(json.dumps({"coordinate_system": coord, "per_model_horizon": pmh}))
    (d / "flags.json").write_text(json.dumps([{"kind": "inaccurate_track", "track_id": 99}]))
    preds = []
    for i, (tid, e) in enumerate(errs):
        preds.append({"id": f"{rid}-{i}", "track_id": tid, "object_class": "person", "t": float(i), "coord": coord,
                      "start": [0, 0], "start_px": [0, 0], "body_px": 100.0,
                      "track_quality": {"tracker_confidence": 0.9}, "observation_window": {"steps": 8},
                      "models": {"learned": {"model_version": model_version, "probability_type": "uncalibrated_weight",
                                             "calibration_status": "uncalibrated", "latency_ms": 1.0}},
                      "evaluations": {"1.0": {"included": True, "scores": {"learned": {"fde_top1" + sfx: e, "fde_top1": e * 100}}}}})
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds))


def test_compare_uses_one_unit_and_pairs_identical_samples(client, tmp_path, monkeypatch):
    import motionpredict.server as srv

    monkeypatch.setattr(srv, "EVAL_DIR", tmp_path)
    _fake_run(tmp_path, "20260101-000000-aaaaaa", "image_pixels", "v1", [(1, 0.3), (2, 0.4), (3, 0.5), (99, 9.0)])
    _fake_run(tmp_path, "20260101-000001-bbbbbb", "image_pixels", "v2", [(1, 0.2), (2, 0.3), (3, 0.4), (99, 0.0)])
    _fake_run(tmp_path, "20260101-000002-cccccc", "image_pixels", "v3", [(1, 0.1)], sha="other")
    r = client.get("/api/eval/compare", params={"runs": "20260101-000000-aaaaaa,20260101-000001-bbbbbb,20260101-000002-cccccc"}).json()
    row = r["runs"][0]
    assert row["unit"] == "body heights"
    pm = row["per_model_horizon"]["learned@1.0"]
    assert pm["fde_top1"]["mean"] == 0.2 and pm["min_fde"]["mean"] == 0.15 and pm["min_ade"]["mean"] == 0.08
    assert len(r["paired"]) == 1  # the run on another source is not paired
    p = r["paired"][0]
    assert p["model"] == "learned" and p["horizon"] == "1.0" and p["n_pairs"] == 3  # flagged track 99 excluded
    assert p["fde_top1"]["mean_diff"] == pytest.approx(-0.1)


def test_explain_mentions_dataset_replay(client):
    result = {"source": "dataset_replay", "tracks": [{"track_id": 1, "state": "confirmed", "age_s": 2.0, "tracker_confidence": 0.9,
                                                      "height": {"status": "unavailable", "reason": "x"}, "forecast": None}],
              "warnings": [], "camera_calibration": {"coordinate_system": "ground_plane_meters"}}
    r = client.post("/api/explain", json={"track_id": 1, "use_llm": False, "result": result}).json()
    assert r["facts"]["data_source"] == "dataset_replay" and "not detections from a live camera" in r["rule_based"]


class _ScriptedDet:
    weights = "scripted"
    is_pose = False
    supports_pose = False

    def __call__(self, frame, **_):
        return []


def test_stream_pause_and_reset_acks(client, monkeypatch):
    import motionpredict.server as srv

    monkeypatch.setattr(srv.S, "detector", _ScriptedDet())
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "pause", "paused": True})
        assert ws.receive_json() == {"type": "ack", "what": "pause", "paused": True}
        ws.send_json({"type": "pause", "paused": False})
        assert ws.receive_json() == {"type": "ack", "what": "pause", "paused": False}
        ws.send_json({"type": "reset"})
        assert ws.receive_json() == {"type": "ack", "what": "reset"}


def test_demo_scenes_training_status(client, monkeypatch):
    from motionpredict import server

    fc = server.S.forecaster
    if fc is not None and fc.available:
        rows = {r["scene"]: r["held_out_for_active_model"] for r in client.get("/api/demo/scenes").json()["scenes"]}
        held = fc.meta.get("held_out_test_scene")
        assert rows[held] is True
        for s in fc.meta.get("trained_on") or []:
            assert rows[s] is False

    class NoModel:
        available = False
        meta: dict = {}

    monkeypatch.setattr(server.S, "forecaster", NoModel())
    rows = client.get("/api/demo/scenes").json()["scenes"]
    assert all(r["held_out_for_active_model"] is None for r in rows)  # unknown, never "in training data"
