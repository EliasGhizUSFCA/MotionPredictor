"""FastAPI server: REST endpoints + WebSocket streaming.

WebSocket ``/ws/stream`` (browser webcam / video file)
    client -> server  binary: [uint32 big-endian header length][header JSON][JPEG bytes]
                      header = {"frame_id": int, "client_ts_ms": float, "capture_ts_ms": float}
    client -> server  text JSON: {"type": "config", ...PipelineConfig}
                                 {"type": "calibration", ...CalibrationRequest} | {"type": "clear_calibration"}
                                 {"type": "server_source", "url": "rtsp://..."}   (needs MP_ALLOW_SERVER_SOURCES=1)
                                 {"type": "reset"}
    server -> client  text JSON: FrameResult | {"type": "hello"|"ack"|"error", ...}

Back-pressure: the receiver keeps only the newest unprocessed frame (older pending frames are
counted as dropped), and the client keeps at most two frames in flight, so latency stays bounded
and predictions are always about the most recent frame.

WebSocket ``/ws/demo?scene=zara1`` - dataset replay (see ``demo.py``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import ValidationError

from .calibration import fit_calibration
from .demo import ReplayDetector, SceneReplay
from .device import select_device
from .evaluation.engine import EVAL_DIR_DEFAULT, EvaluationRun
from .evaluation.report import load_predictions, summarise
from .explain import facts_for_track, llm_summary, rule_based_summary
from .forecasting.service import MODELS_DIR, Forecaster
from .pipeline import Session, now_ms
from .schemas import SCHEMA_VERSION, CalibrationRequest, ExplainRequest, PipelineConfig

log = logging.getLogger("motionpredict")
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("MP_DATA_DIR", ROOT / "data" / "raw"))
EVAL_DIR = Path(os.environ.get("MP_EVAL_DIR", EVAL_DIR_DEFAULT))
UPLOAD_DIR = EVAL_DIR / "_uploads"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class State:
    device_info: dict = {}
    detector = None
    detector_error: str | None = None
    forecaster: Forecaster | None = None
    detector_lock = threading.Lock()
    runs: dict[str, EvaluationRun] = {}


S = State()
app = FastAPI(title="MotionPredict AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MP_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class LockedDetector:
    """Serialises access to the shared detector across sessions."""

    def __init__(self, det):
        self._d = det
        self.weights = det.weights
        self.is_pose = det.is_pose

    def __call__(self, *a, **k):
        with S.detector_lock:
            return self._d(*a, **k)


def init_models(load_detector: bool = True) -> None:
    S.device_info = select_device()
    dev = S.device_info["device"]
    S.forecaster = Forecaster(device=dev)
    if load_detector:
        try:
            from .detection import PersonDetector

            S.detector = LockedDetector(PersonDetector(device=dev))
        except Exception as e:  # surfaced in /api/health and in stream errors
            S.detector_error = f"{type(e).__name__}: {e}"
            log.exception("detector failed to load")


@app.on_event("startup")
async def _startup() -> None:
    if os.environ.get("MP_SKIP_MODEL_LOAD") != "1":
        await asyncio.to_thread(init_models)
    else:
        S.device_info = select_device()
        S.forecaster = Forecaster(device=S.device_info["device"])


# ---------------------------------------------------------------- REST
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "device": S.device_info,
        "detector": {"loaded": S.detector is not None, "weights": getattr(S.detector, "weights", None), "error": S.detector_error},
        "forecaster": S.forecaster.info() if S.forecaster else None,
        "llm": {"configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
                "model": os.environ.get("MP_LLM_MODEL", "claude-opus-5-5")},
        "demo_available": all((RAW_DIR / f).exists() for f in ("crowds_zara01.txt",)),
        "server_sources_allowed": os.environ.get("MP_ALLOW_SERVER_SOURCES") == "1",
        "privacy": {"frames_stored": False, "face_recognition": False, "ids": "session-local, cleared on stop"},
    }


@app.get("/api/model")
def model_card() -> dict:
    bench = MODELS_DIR / "benchmark.json"
    return {
        "forecaster": S.forecaster.info() if S.forecaster else None,
        "report": S.forecaster.report if S.forecaster else None,
        "benchmark": json.loads(bench.read_text()) if bench.exists() else None,
    }


@app.get("/api/config/defaults")
def config_defaults() -> dict:
    return PipelineConfig().model_dump()


@app.post("/api/calibration/fit")
def calibration_fit(req: CalibrationRequest) -> dict:
    try:
        cal = fit_calibration(req.image_points, req.world_points, req.camera_height_m, req.reference, req.height_tolerance)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return cal.to_dict()


@app.post("/api/explain")
def explain(req: ExplainRequest) -> dict:
    facts = facts_for_track(req.result, req.track_id)
    if facts is None:
        return {"ok": False, "error": "no confirmed track to explain"}
    out = {"ok": True, "facts": facts, "rule_based": rule_based_summary(facts)}
    if req.use_llm:
        out["llm"] = llm_summary(facts)
    return out


@app.get("/api/demo/scenes")
def demo_scenes() -> dict:
    from .forecasting.data import SCENES

    held = (S.forecaster.meta.get("held_out_test_scene") if S.forecaster else None)
    return {"scenes": [{"scene": s, "available": all((RAW_DIR / f).exists() for f in fs),
                        "held_out_for_active_model": s == held} for s, fs in SCENES.items()]}


# ------------------------------------------------------- evaluation API
def _run_dir(run_id: str) -> Path:
    d = (EVAL_DIR / run_id).resolve()
    if EVAL_DIR.resolve() not in d.parents or not d.exists():
        raise HTTPException(404, "unknown run")
    return d


def _forecaster_for(checkpoint: str | None) -> Forecaster:
    if not checkpoint:
        return S.forecaster
    p = (MODELS_DIR / checkpoint / "model.pt").resolve()
    if MODELS_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(400, f"unknown checkpoint {checkpoint}")
    return Forecaster(str(p), device=S.device_info.get("device", "cpu"))


def _start_run(cfg: dict, video: Path | None, scene: str | None, keep_video: bool) -> dict:
    fc = _forecaster_for(cfg.get("checkpoint"))
    if video is not None and S.detector is None:
        raise HTTPException(503, S.detector_error or "detector not loaded")
    run = EvaluationRun(EVAL_DIR)
    run.dir.mkdir(parents=True, exist_ok=True)
    (run.dir / "request.json").write_text(json.dumps({"config": cfg, "scene": scene, "keep_video": keep_video,
                                                       "video_kept_at": str(video) if (video and keep_video) else None}))
    S.runs[run.id] = run
    threading.Thread(target=run.run, args=(cfg, S.detector, fc), kwargs={"video_path": video, "scene": scene, "raw_dir": RAW_DIR,
                                                                         "keep_video": keep_video}, daemon=True).start()
    return {"run_id": run.id}


@app.post("/api/eval/runs")
async def eval_upload(file: UploadFile = File(...), config: str = Form("{}"), keep_video: bool = Form(False)) -> dict:
    """Upload a video for whole-recording prediction-vs-reality evaluation (processed locally)."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {ext}; use one of {sorted(VIDEO_EXT)}")
    cfg = json.loads(config or "{}")
    cfg["filename"] = Path(file.filename).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{int(time.time() * 1000)}{ext}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return _start_run(cfg, dest, None, keep_video)


@app.post("/api/eval/runs/dataset")
def eval_dataset(body: dict) -> dict:
    """Evaluate on a human-annotated ETH/UCY scene (true ground truth; validates the engine)."""
    scene = body.get("scene", "zara1")
    return _start_run(body.get("config", {}), None, scene, False)


@app.post("/api/eval/runs/{run_id}/rerun")
def eval_rerun(run_id: str, body: dict) -> dict:
    """Re-run an earlier evaluation (same source + config) with another model checkpoint."""
    d = _run_dir(run_id)
    req = json.loads((d / "request.json").read_text())
    cfg = {**req["config"], "checkpoint": body.get("checkpoint"), "rerun_of": run_id}
    if req["scene"]:
        return _start_run(cfg, None, req["scene"], False)
    if not req.get("video_kept_at") or not Path(req["video_kept_at"]).exists():
        raise HTTPException(409, "original video was deleted after analysis (privacy default); upload it again or enable 'keep video'")
    return _start_run(cfg, Path(req["video_kept_at"]), None, True)


@app.get("/api/eval/runs")
def eval_list() -> dict:
    out = []
    if EVAL_DIR.exists():
        for d in sorted(EVAL_DIR.iterdir(), reverse=True):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            out.append(_run_status(d))
    return {"runs": out}


def _run_status(d: Path) -> dict:
    run = S.runs.get(d.name)
    st = {"status": run.status, "progress": run.progress, "error": run.error} if run else (
        json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"status": "unknown"})
    meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
    req = json.loads((d / "request.json").read_text()) if (d / "request.json").exists() else {}
    return {"run_id": d.name, **st, "meta": meta, "request": req}


@app.get("/api/eval/runs/{run_id}")
def eval_get(run_id: str) -> dict:
    return _run_status(_run_dir(run_id))


@app.delete("/api/eval/runs/{run_id}")
def eval_delete(run_id: str) -> dict:
    import shutil

    d = _run_dir(run_id)
    run = S.runs.pop(run_id, None)
    if run:
        run.cancel.set()
    shutil.rmtree(d)
    return {"deleted": run_id}


@app.get("/api/eval/runs/{run_id}/summary")
def eval_summary(run_id: str) -> dict:
    d = _run_dir(run_id)
    if not (d / "summary.json").exists():
        raise HTTPException(409, "run not finished")
    return json.loads((d / "summary.json").read_text())


@app.get("/api/eval/runs/{run_id}/predictions")
def eval_predictions(run_id: str, track_id: int | None = None, t_from: float | None = None, t_to: float | None = None) -> dict:
    preds = load_predictions(_run_dir(run_id))
    if track_id is not None:
        preds = [p for p in preds if p["track_id"] == track_id]
    if t_from is not None:
        preds = [p for p in preds if p["t"] >= t_from]
    if t_to is not None:
        preds = [p for p in preds if p["t"] <= t_to]
    return {"predictions": preds}


@app.get("/api/eval/runs/{run_id}/tracks")
def eval_tracks(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "tracks.json").read_text())


@app.get("/api/eval/runs/{run_id}/export.csv")
def eval_csv(run_id: str) -> PlainTextResponse:
    d = _run_dir(run_id)
    return PlainTextResponse((d / "predictions.csv").read_text(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="motionpredict-{run_id}.csv"'})


@app.get("/api/eval/runs/{run_id}/export.json")
def eval_json(run_id: str) -> dict:
    d = _run_dir(run_id)
    return {"run": json.loads((d / "run.json").read_text()), "summary": json.loads((d / "summary.json").read_text()),
            "predictions": load_predictions(d), "flags": json.loads((d / "flags.json").read_text())}


@app.get("/api/eval/runs/{run_id}/report.html")
def eval_report(run_id: str) -> HTMLResponse:
    return HTMLResponse((_run_dir(run_id) / "report.html").read_text())


@app.post("/api/eval/runs/{run_id}/flags")
def eval_flag(run_id: str, body: dict) -> dict:
    """Review workflow: flag a prediction (hard case) or a track (inaccurate tracking), with an optional note
    or corrected future points. 'inaccurate_track' flags exclude that track when the summary is recomputed."""
    d = _run_dir(run_id)
    kind = body.get("kind")
    if kind not in ("hard_case", "inaccurate_track", "annotation_correction", "unflag"):
        raise HTTPException(400, "kind must be hard_case | inaccurate_track | annotation_correction | unflag")
    flags = json.loads((d / "flags.json").read_text())
    if kind == "unflag":
        flags = [f for f in flags if not (f.get("prediction_id") == body.get("prediction_id") and f.get("track_id") == body.get("track_id"))]
    else:
        flags.append({"kind": kind, "prediction_id": body.get("prediction_id"), "track_id": body.get("track_id"),
                      "note": str(body.get("note", ""))[:500], "corrected_future": body.get("corrected_future"),
                      "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (d / "flags.json").write_text(json.dumps(flags, indent=1))
    summarise(d)
    return {"flags": flags}


@app.get("/api/eval/runs/{run_id}/review_set")
def eval_review_set(run_id: str, top_n: int = 50) -> dict:
    """Export flagged + highest-error cases as training/review examples.

    Each example holds only information available at prediction time (observation window) and the
    observed future. Use it as a *separate* scene/split: never mix these samples into the test set
    you report on, and never evaluate on the same recording you fine-tune on."""
    d = _run_dir(run_id)
    summ = json.loads((d / "summary.json").read_text())
    flags = json.loads((d / "flags.json").read_text())
    ids = {f["prediction_id"] for f in flags if f.get("prediction_id")} | {c["prediction_id"] for c in summ["failure_cases"][:top_n]}
    bad_tracks = {f["track_id"] for f in flags if f["kind"] == "inaccurate_track"}
    corr = {f["prediction_id"]: f["corrected_future"] for f in flags if f["kind"] == "annotation_correction" and f.get("corrected_future")}
    ex = []
    for p in load_predictions(d):
        if p["id"] in ids and p["track_id"] not in bad_tracks:
            ex.append({"prediction_id": p["id"], "t": p["t"], "coord": p["coord"], "observation_window": p["observation_window"],
                       "future": corr.get(p["id"], p["actual_future"]), "future_source": "user_correction" if p["id"] in corr else "tracker",
                       "flags": p["flags"]})
    return {"run_id": run_id, "usage_note": "Review/fine-tuning set. Keep it in its own split; do not report test accuracy on these samples.",
            "source_recording_sha256": json.loads((d / "run.json").read_text())["source"].get("sha256"), "examples": ex}


@app.get("/api/eval/compare")
def eval_compare(runs: str) -> dict:
    """Cross-run comparison (e.g. model versions, sports, camera perspectives via run tags)."""
    out = []
    for rid in runs.split(","):
        d = _run_dir(rid.strip())
        if (d / "summary.json").exists():
            m = json.loads((d / "run.json").read_text())
            sm = json.loads((d / "summary.json").read_text())
            out.append({"run_id": rid, "tags": m.get("tags"), "models": m.get("models"), "coordinate_system": sm["coordinate_system"],
                        "per_model_horizon": {k: {kk: v.get(kk) for kk in ("n", "ade_top1", "fde_top1", "ade_top1_bh", "fde_top1_bh", "min_fde",
                                                                            "miss_rate_top1", "coverage90")} for k, v in sm["per_model_horizon"].items()}})
    return {"runs": out}


@app.get("/api/models")
def list_models() -> dict:
    out = []
    for d in sorted(MODELS_DIR.glob("socialmdn_*")):
        if (d / "model.pt").exists():
            out.append({"checkpoint": d.name, "active": S.forecaster is not None and Path(str(d / "model.pt")).resolve()
                        == Path(os.environ.get("MP_FORECAST_CKPT", MODELS_DIR / "socialmdn_zara1" / "model.pt")).resolve()})
    return {"checkpoints": out}


# ----------------------------------------------------------- WebSocket
def decode_frame(payload: bytes) -> tuple[dict, np.ndarray, float]:
    t0 = time.perf_counter()
    (hlen,) = struct.unpack(">I", payload[:4])
    header = json.loads(payload[4 : 4 + hlen])
    img = cv2.imdecode(np.frombuffer(payload[4 + hlen :], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode JPEG")
    return header, img, (time.perf_counter() - t0) * 1000


async def handle_control(ws: WebSocket, sess: Session, msg: dict) -> None:
    typ = msg.get("type")
    if typ == "config":
        try:
            cfg = PipelineConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await ws.send_json({"type": "error", "where": "config", "error": e.errors(include_url=False, include_context=False)})
            return
        sess.apply_config(cfg)
        await ws.send_json({"type": "ack", "what": "config", "config": cfg.model_dump()})
    elif typ == "calibration":
        try:
            req = CalibrationRequest(**{k: v for k, v in msg.items() if k != "type"})
            sess.calibration = fit_calibration(req.image_points, req.world_points, req.camera_height_m, req.reference, req.height_tolerance)
            for st in sess.kin.values():
                st["w"] = None
                st["heights"].clear()
            await ws.send_json({"type": "ack", "what": "calibration", "calibration": sess.calibration.to_dict()})
        except (ValueError, ValidationError) as e:
            await ws.send_json({"type": "error", "where": "calibration", "error": str(e)})
    elif typ == "clear_calibration":
        sess.calibration = None
        await ws.send_json({"type": "ack", "what": "clear_calibration"})
    elif typ == "reset":
        sess.reset()
        await ws.send_json({"type": "ack", "what": "reset"})


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    if S.detector is None:
        await ws.send_json({"type": "error", "where": "detector", "error": S.detector_error or "detector not loaded"})
        await ws.close()
        return
    sess = Session(S.detector, S.forecaster, S.device_info.get("device", "cpu"))
    await ws.send_json({"type": "hello", "session_id": sess.id, "schema_version": SCHEMA_VERSION, "config": sess.config.model_dump()})
    slot: dict = {"item": None}
    ev = asyncio.Event()
    closed = asyncio.Event()
    server_src: dict = {"task": None}

    async def worker() -> None:
        while not closed.is_set():
            await ev.wait()
            ev.clear()
            item, slot["item"] = slot["item"], None
            if item is None:
                continue
            header, img, decode_ms, nbytes, recv_ms, src, jpeg_b64 = item
            try:
                res = await asyncio.to_thread(
                    sess.process, img, int(header.get("frame_id", 0)), header.get("capture_ts_ms", recv_ms) / 1000.0,
                    recv_ms, header.get("client_ts_ms"), src, decode_ms, nbytes,
                )
                if jpeg_b64:
                    res.frame_jpeg_b64 = jpeg_b64
                await ws.send_text(res.model_dump_json())
            except WebSocketDisconnect:
                return
            except Exception as e:  # report and keep serving
                log.exception("processing failed")
                try:
                    await ws.send_json({"type": "error", "where": "pipeline", "error": f"{type(e).__name__}: {e}"})
                except Exception:
                    return

    def enqueue(item) -> None:
        sess.frames_received += 1
        if slot["item"] is not None:
            sess.frames_dropped += 1
        slot["item"] = item
        ev.set()

    async def server_source(url: str) -> None:
        """Pull frames from an RTSP/HTTP/file source on the server (local, authorised setups only)."""
        cap = await asyncio.to_thread(cv2.VideoCapture, url)
        fid = 0
        try:
            while not closed.is_set():
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    await ws.send_json({"type": "error", "where": "server_source", "error": "source ended or disconnected; retrying in 2s"})
                    await asyncio.sleep(2)
                    cap.release()
                    cap = await asyncio.to_thread(cv2.VideoCapture, url)
                    continue
                fid += 1
                t = now_ms()
                okj, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                enqueue(({"frame_id": fid, "capture_ts_ms": t}, frame, 0.0, len(jpg), t, "server",
                         base64.b64encode(jpg.tobytes()).decode() if okj else None))
                await asyncio.sleep(0.001)
        finally:
            cap.release()

    wtask = asyncio.create_task(worker())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                recv = now_ms()
                try:
                    header, img, dms = decode_frame(msg["bytes"])
                except Exception as e:
                    await ws.send_json({"type": "error", "where": "decode", "error": str(e)})
                    continue
                enqueue((header, img, dms, len(msg["bytes"]), recv, "browser", None))
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "server_source":
                    if os.environ.get("MP_ALLOW_SERVER_SOURCES") != "1":
                        await ws.send_json({"type": "error", "where": "server_source",
                                            "error": "server-side sources disabled; set MP_ALLOW_SERVER_SOURCES=1 for an authorised local setup"})
                        continue
                    if server_src["task"]:
                        server_src["task"].cancel()
                    sess.reset()
                    server_src["task"] = asyncio.create_task(server_source(str(data.get("url", ""))))
                    await ws.send_json({"type": "ack", "what": "server_source"})
                elif data.get("type") == "stop_server_source":
                    if server_src["task"]:
                        server_src["task"].cancel()
                        server_src["task"] = None
                    await ws.send_json({"type": "ack", "what": "stop_server_source"})
                else:
                    await handle_control(ws, sess, data)
    except WebSocketDisconnect:
        pass
    finally:
        closed.set()
        ev.set()
        if server_src["task"]:
            server_src["task"].cancel()
        wtask.cancel()
        sess.reset()  # privacy: drop all per-person buffers immediately


@app.websocket("/ws/demo")
async def ws_demo(ws: WebSocket, scene: str = "zara1", speed: float = 1.0) -> None:
    await ws.accept()
    try:
        replay = SceneReplay(RAW_DIR, scene)
    except (FileNotFoundError, KeyError) as e:
        await ws.send_json({"type": "error", "where": "demo", "error": str(e)})
        await ws.close()
        return
    det = ReplayDetector()
    sess = Session(det, S.forecaster, S.device_info.get("device", "cpu"))
    sess.calibration = replay.calibration
    cfg = sess.config.model_copy(update={"pose": False, "camera_motion_check": False, "det_conf": 0.5})
    sess.apply_config(cfg)
    await ws.send_json({"type": "hello", "session_id": sess.id, "schema_version": SCHEMA_VERSION, "config": cfg.model_dump(),
                        "demo": {"scene": scene, "file": replay.file, "duration_s": replay.duration,
                                 "held_out_for_active_model": S.forecaster.meta.get("held_out_test_scene") == scene}})
    state = {"paused": False, "speed": max(0.25, min(speed, 4.0))}

    async def receiver() -> None:
        while True:
            data = json.loads(await ws.receive_text())
            if data.get("type") == "pause":
                state["paused"] = bool(data.get("paused", True))
            elif data.get("type") == "speed":
                state["speed"] = max(0.25, min(float(data.get("speed", 1.0)), 4.0))
            elif data.get("type") == "config":
                data.update({"pose": False, "camera_motion_check": False})
            if data.get("type") in ("config", "reset"):
                await handle_control(ws, sess, data)

    rtask = asyncio.create_task(receiver())
    fid = 0
    try:
        while not rtask.done():
            if state["paused"]:
                await asyncio.sleep(0.1)
                continue
            t_loop = time.perf_counter()
            frame, t, mapping = replay.step(det)
            fid += 1
            recv = now_ms()
            res = await asyncio.to_thread(sess.process, frame, fid, t, recv, None, "dataset_replay", 0.0, 0)
            # attach the true future for each track (matched by its detection box)
            gt = {}
            for tr in sess.tracker.tracks:
                if tr.last_det_box is None or not tr.matched_this_frame:
                    continue
                for pid, box in mapping.items():
                    if np.allclose(tr.last_det_box, box):
                        gt[str(tr.track_id)] = replay.future_of(pid, t)
                        break
            res.replay_ground_truth = gt
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            res.frame_jpeg_b64 = base64.b64encode(jpg.tobytes()).decode() if ok else None
            await ws.send_text(res.model_dump_json())
            await asyncio.sleep(max(0.0, replay.tick_s / state["speed"] - (time.perf_counter() - t_loop)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rtask.cancel()
        sess.reset()


FRONTEND_OUT = ROOT / "frontend" / "out"
if FRONTEND_OUT.exists():  # serve the static export of the UI from the same process (optional)
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=FRONTEND_OUT, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run("motionpredict.server:app", host=os.environ.get("MP_HOST", "127.0.0.1"), port=int(os.environ.get("MP_PORT", "8000")))


if __name__ == "__main__":
    main()
