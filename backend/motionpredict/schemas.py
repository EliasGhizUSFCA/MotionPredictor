"""API / WebSocket data contracts. Mirrored by frontend/src/lib/types.ts - keep both in sync.

Coordinate systems
* ``image_pixels``        - pixel coordinates of the processed frame (origin top-left, y down).
* ``ground_plane_meters`` - metric ground plane defined by the user's calibration (homography).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

CoordinateSystem = Literal["image_pixels", "ground_plane_meters"]
PredictionSource = Literal["trained_model", "physics_baseline"]
ProbabilityType = Literal["calibrated_probability", "uncalibrated_weight", "none"]
TrackState = Literal["tentative", "confirmed", "lost"]


class StageEvent(BaseModel):
    stage: str
    label: str
    t_start_ms: float  # server wall-clock, epoch ms
    duration_ms: float
    status: Literal["ok", "skipped", "error"] = "ok"
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    explanation: str = ""
    error: str | None = None


class Detection(BaseModel):
    bbox: list[float]  # x1,y1,x2,y2 pixels
    confidence: float
    keypoints: list[list[float]] | None = None  # 17 x [x, y, conf] (COCO order)


class HeightEstimate(BaseModel):
    status: Literal["calibrated", "uncalibrated", "unavailable"]
    meters: float | None = None
    interval_m: list[float] | None = None  # [low, high] approximate 90% interval
    relative_image_height: float | None = None  # bbox height / frame height
    reason: str = ""


class Kinematics(BaseModel):
    units: Literal["px", "m"]
    position: list[float]
    velocity: list[float]
    speed: float
    heading_deg: float  # atan2(vy, vx) in the track's coordinate system
    acceleration: list[float]
    accel_magnitude: float
    turn_rate_deg_s: float
    movement_state: str
    speed_category: str | None = None  # only in calibrated (meters) mode
    position_std: float


class Trajectory(BaseModel):
    id: str  # stable mode label (A..E) or CV/CA for baselines
    rank: int
    probability: float | None  # None for physics baseline (no route probabilities)
    times: list[float]  # seconds ahead of the last observation
    points: list[list[float]]  # in ``coordinate_system``
    image_points: list[list[float]] | None = None  # projected for overlay when calibrated
    sigma: list[list[float]]  # per point [sigma_major, sigma_minor, angle_rad] (1-sigma ellipse)
    band_left: list[list[float]] | None = None  # image-space 2-sigma cross-track boundary
    band_right: list[list[float]] | None = None


class Forecast(BaseModel):
    track_id: int
    horizon_seconds: float
    coordinate_system: CoordinateSystem
    prediction_source: PredictionSource
    model_version: str
    probability_type: ProbabilityType
    calibration_status: str  # probability calibration status
    trajectories: list[Trajectory]
    residual_probability: float | None = None  # mass of modes not shown
    history_steps_used: int
    warnings: list[str] = Field(default_factory=list)


class TrackOut(BaseModel):
    track_id: int
    state: TrackState
    age_s: float
    hits: int
    time_since_update_s: float
    tracker_confidence: float
    bbox: list[float]
    det_confidence: float
    footpoint_px: list[float]
    keypoints: list[list[float]] | None = None
    history_px: list[list[float]]  # [x, y, t_rel] observed (filtered) footpoints
    history_world: list[list[float]] | None = None
    kinematics: Kinematics | None = None
    kinematics_world: Kinematics | None = None
    height: HeightEstimate
    forecast: Forecast | None = None


class FrameResult(BaseModel):
    type: Literal["result"] = "result"
    schema_version: str = SCHEMA_VERSION
    session_id: str
    frame_id: int
    source: Literal["browser", "server", "dataset_replay"]
    client_ts_ms: float | None = None
    server_received_ms: float
    server_done_ms: float
    frame_width: int
    frame_height: int
    processing_ms: float
    tracks: list[TrackOut]
    detections: int
    events: list[StageEvent]
    warnings: list[str]
    camera_calibration: dict[str, Any]
    camera_motion: dict[str, Any] | None = None
    stats: dict[str, Any]
    replay_ground_truth: dict[str, list[list[float]]] | None = None
    frame_jpeg_b64: str | None = None


class PipelineConfig(BaseModel):
    det_conf: float = Field(0.35, ge=0.05, le=0.95)
    det_iou: float = Field(0.5, ge=0.1, le=0.95)
    imgsz: int = Field(640, ge=160, le=1280)
    pose: bool = True
    pose_every: int = Field(3, ge=1, le=30)  # run the pose model on every Nth processed frame
    forecaster: Literal["learned", "baseline"] = "learned"
    horizon_s: float = Field(3.0, ge=0.5, le=5.0)
    num_hypotheses: int = Field(3, ge=1, le=5)
    history_s: float = Field(6.0, ge=1.0, le=30.0)
    track_buffer_s: float = Field(1.0, ge=0.2, le=5.0)
    camera_motion_check: bool = True


class CalibrationRequest(BaseModel):
    """Image points are normalised [0,1] coordinates of the frame (resolution independent)."""

    image_points: list[list[float]]
    world_points: list[list[float]]  # meters on the ground plane
    camera_height_m: float | None = None
    reference: dict[str, Any] | None = None  # {"foot":[u,v], "top":[u,v], "height_m": float}
    height_tolerance: float = 0.05


class ExplainRequest(BaseModel):
    track_id: int | None = None
    use_llm: bool = False
    result: dict[str, Any]
