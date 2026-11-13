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
