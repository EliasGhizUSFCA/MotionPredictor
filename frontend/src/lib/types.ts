// Mirrors backend/motionpredict/schemas.py (SCHEMA_VERSION 1.0). Keep both in sync.

export type CoordinateSystem = "image_pixels" | "ground_plane_meters";
export type PredictionSource = "trained_model" | "physics_baseline";
export type ProbabilityType = "calibrated_probability" | "uncalibrated_weight" | "none";
export type TrackState = "tentative" | "confirmed" | "lost";

export interface StageEvent {
  stage: string;
  label: string;
  t_start_ms: number;
  duration_ms: number;
  status: "ok" | "skipped" | "error";
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  confidence: number | null;
  explanation: string;
  error: string | null;
}

export interface HeightEstimate {
  status: "calibrated" | "uncalibrated" | "unavailable";
  meters: number | null;
  interval_m: [number, number] | null;
  relative_image_height: number | null;
  reason: string;
}

export interface Kinematics {
  units: "px" | "m";
  position: [number, number];
  velocity: [number, number];
  speed: number;
  heading_deg: number;
  acceleration: [number, number];
  accel_magnitude: number;
  turn_rate_deg_s: number;
  movement_state: string;
  speed_category: string | null;
  position_std: number;
}

export type Pt = [number, number];
export type MaybePt = [number | null, number | null];

export interface Trajectory {
  id: string;
  rank: number;
  probability: number | null;
  times: number[];
  points: Pt[];
  image_points: MaybePt[] | null;
  sigma: [number, number, number][];
  band_left: MaybePt[] | null;
  band_right: MaybePt[] | null;
}

export interface Forecast {
  track_id: number;
  horizon_seconds: number;
  coordinate_system: CoordinateSystem;
  prediction_source: PredictionSource;
  model_version: string;
  probability_type: ProbabilityType;
  calibration_status: string;
  trajectories: Trajectory[];
  residual_probability: number | null;
  history_steps_used: number;
  warnings: string[];
}

export interface TrackOut {
  track_id: number;
  state: TrackState;
  age_s: number;
  hits: number;
  time_since_update_s: number;
  tracker_confidence: number;
  bbox: [number, number, number, number];
  det_confidence: number;
  footpoint_px: Pt;
  keypoints: [number, number, number][] | null;
  history_px: [number, number, number][];
  history_world: [number, number, number][] | null;
  kinematics: Kinematics | null;
  kinematics_world: Kinematics | null;
  height: HeightEstimate;
  forecast: Forecast | null;
}

export interface CameraCalibration {
  status: string;
  coordinate_system: CoordinateSystem;
  homography?: number[][];
  homography_inv?: number[][];
  horizon_v?: [number | null, number | null];
  rms_error_m?: number | null;
  camera_height_m?: number | null;
  camera_height_source?: string | null;
  height_estimation?: boolean;
  warnings?: string[];
  image_points?: number[][];
  world_points?: number[][];
}

export interface FrameResult {
  type: "result";
  schema_version: string;
  session_id: string;
  frame_id: number;
  source: "browser" | "server" | "dataset_replay";
  client_ts_ms: number | null;
  server_received_ms: number;
  server_done_ms: number;
  frame_width: number;
  frame_height: number;
  processing_ms: number;
  tracks: TrackOut[];
  detections: number;
  events: StageEvent[];
  warnings: string[];
  camera_calibration: CameraCalibration;
  camera_motion: { available: boolean; shift_px: Pt; shift_px_per_s: number; camera_moving: boolean; tracked_points: number } | null;
  stats: {
    frames_received: number;
    frames_processed: number;
    frames_dropped: number;
    input_fps: number | null;
    inference_fps: number | null;
    mean_processing_ms: number;
    compute_ms_mean?: number | null; // worker start -> done (excludes queue wait); inference_fps = 1000 / compute_ms_mean
    queue_ms_mean?: number | null; // receipt -> worker start
    device: string;
    forecaster_active: PredictionSource | null;
  };
  replay_ground_truth: Record<string, [number, number, number][]> | null;
  frame_jpeg_b64: string | null;
}

export interface PipelineConfig {
  det_conf: number;
  det_iou: number;
  imgsz: number;
  pose: boolean;
  pose_every: number;
  forecaster: "learned" | "baseline";
  horizon_s: number;
  num_hypotheses: number;
  history_s: number;
  track_buffer_s: number;
  camera_motion_check: boolean;
}

export const DEFAULT_CONFIG: PipelineConfig = {
  det_conf: 0.35,
  det_iou: 0.5,
  imgsz: 640,
  pose: true,
  pose_every: 3,
  forecaster: "learned",
  horizon_s: 3,
  num_hypotheses: 3,
  history_s: 6,
  track_buffer_s: 1,
  camera_motion_check: true,
};

export interface CalibrationRequest {
  image_points: number[][];
  world_points: number[][];
  camera_height_m?: number | null;
  reference?: { foot: number[]; top: number[]; height_m: number } | null;
  height_tolerance?: number;
}

export interface Health {
  status: string;
  schema_version: string;
  device: { device: string; device_name: string; cuda_available: boolean; mps_available: boolean; torch_version: string; notes: string[] };
  detector: { loaded: boolean; weights: string | null; error: string | null };
  forecaster: {
    available: boolean;
    load_error: string | null;
    version: string | null;
    probability_calibration_validated: boolean;
    probability_gate?: { passed: boolean; reasons: string[]; test_ece_all_modes: number | null; test_ece_top_mode: number | null; requires: string };
    ece_threshold: number;
    ece_top_mode_threshold?: number;
    meta: Record<string, any>;
  } | null;
  llm: { configured: boolean; model: string };
  demo_available: boolean;
  server_sources_allowed: boolean;
}

// ---------------------------------------------------------------- evaluation
export interface EvalModelOut {
  prediction_source: PredictionSource;
  model_version: string;
  probability_type: ProbabilityType;
  calibration_status: string;
  trajectories: Trajectory[];
  history_steps_used: number;
  latency_ms: number;
}

export interface EvalScore {
  ade_top1: number;
  fde_top1: number;
  min_ade: number;
  min_fde: number;
  ade_top1_bh?: number;
  fde_top1_bh?: number;
  min_ade_bh?: number;
  min_fde_bh?: number;
  w_ade: number | null;
  w_fde?: number | null;
  w_ade_bh?: number | null;
  w_fde_bh?: number | null;
  miss_top1: boolean;
  miss_k: boolean;
  nll_final: number;
  in_hdr50: boolean;
  in_hdr90: boolean;
  top_id: string;
  top_prob: number | null;
  best_id: string;
  k: number;
  mode_probs: Record<string, number> | null;
  miss_threshold: number;
}

export interface EvalPrediction {
  id: string;
  track_id: number;
  object_class: string;
  t: number;
  frame: number;
  coord: CoordinateSystem;
  hmax: number;
  horizons: number[];
  body_px: number;
  start_px: Pt;
  start: Pt;
  observation_window: { dt_s: number; steps: number; positions: Pt[] };
  history_px: [number, number, number][];
  track_quality: { age_s: number; hits: number; tracker_confidence: number; det_confidence: number };
  camera_moving_at_t: boolean;
  causality: { latest_frame_t_used: number };
  models: Record<string, EvalModelOut>;
  evaluations: Record<string, { included: boolean; exclusion_reasons?: string[]; scores?: Record<string, EvalScore> }>;
  flags: string[];
  actual_future: [number, number, number][]; // [t_rel, x, y] in coord units
}

export interface Stat {
  mean: number;
  median: number;
  p50?: number;
  p90: number;
  n: number;
}

export interface ModelHorizonSummary {
  n: number;
  ade_top1: Stat | null;
  fde_top1: Stat | null;
  min_ade: Stat | null;
  min_fde: Stat | null;
  w_ade: Stat | null;
  w_fde?: Stat | null;
  nll_final: Stat | null;
  ade_top1_bh?: Stat | null;
  fde_top1_bh?: Stat | null;
  min_ade_bh?: Stat | null;
  min_fde_bh?: Stat | null;
  w_ade_bh?: Stat | null;
  w_fde_bh?: Stat | null;
  forecast_failure_rate?: number | null;
  miss_rate_top1?: number;
  miss_rate_k?: number;
  coverage50?: number;
  coverage90?: number;
  latency_ms?: number;
  k?: number;
  probability_type?: ProbabilityType;
  calibration_status?: string;
  reliability?: { available: boolean; ece?: number; n_mode_predictions?: number; bins?: { lo: number; hi: number; n: number; confidence: number; frequency: number }[] };
}

export interface EvalSummary {
  run_id: string;
  coordinate_system: CoordinateSystem;
  unit: "m" | "px";
  normalised_unit: string | null;
  ground_truth: string;
  models: string[];
  horizons: number[];
  n_predictions: number;
  inclusion: Record<string, Record<string, number>>; // per horizon key; may include "forecast_failed", "flagged_inaccurate_track"
  flagged_tracks_excluded: number[]; // excluded from EVERY aggregate
  forecast_failures?: { count: number; by_reason: Record<string, number>; by_model?: Record<string, number>; n_eligible?: number };
  scene_in_active_model_training?: boolean | null; // dataset runs; null for video / unknown
  per_model_horizon: Record<string, ModelHorizonSummary>;
  paired_comparison: Record<string, { comparison: string; unit: string; ade_top1: { mean_diff: number; ci95: [number, number]; n_pairs: number }; fde_top1: { mean_diff: number; ci95: [number, number]; n_pairs: number } }>;
  per_track: { model: string; track_id: number; n: number; mean_fde: number }[];
  failure_cases: Record<string, any>[];
  best_cases: Record<string, any>[];
  selected_horizon_for_rankings: number;
  ranking_metric: string;
  tracking_over_time: { t: number; frame: number; tracks: number; detections: number; camera_moving: boolean; cut: boolean; mean_tracker_conf: number | null; processing_ms: number }[];
  heatmap: {
    grid: { nx: number; ny: number; x0: number; x1: number; y0: number; y1: number; space: string };
    unit: string;
    min_samples_to_display: number;
    cells: { model: string; horizon: number; i: number; j: number; n: number; n_total: number; mean_fde: number | null; exclusion_rate: number | null; mean_turn_deg: number | null }[];
  };
  metric_definitions: Record<string, string>;
}

export interface EvalRunInfo {
  run_id: string;
  status: string;
  progress: number;
  error: string | null;
  meta: Record<string, any> | null;
  request: Record<string, any>;
}

export interface EvalFlag {
  kind: "hard_case" | "inaccurate_track" | "annotation_correction";
  prediction_id: string | null;
  track_id: number | null;
  note: string;
  corrected_future: [number, number, number][] | null; // [t_rel, x, y] in the run coordinate system
  created: string;
}

// GET /api/eval/compare?runs=a,b - every distance in a run row uses that row's `unit`.
export interface EvalCompareRow {
  n: number;
  ade_top1: Stat | null;
  fde_top1: Stat | null;
  min_ade: Stat | null;
  min_fde: Stat | null;
  miss_rate_top1: number | null;
  coverage90: number | null;
  coverage50?: number | null;
  latency_ms?: number | null;
  reliability_ece?: number | null;
}

export interface EvalCompareRun {
  run_id: string;
  tags: { sport?: string | null; camera_perspective?: string | null; notes?: string | null } | null;
  models: Record<string, any> | null;
  coordinate_system: CoordinateSystem;
  unit: "m" | "body heights";
  source: Record<string, any> | null;
  per_model_horizon: Record<string, EvalCompareRow>;
}

export interface EvalComparePaired {
  run_a: string;
  run_b: string;
  model: string;
  horizon: string;
  n_pairs: number;
  unit?: string;
  comparison?: string;
  fde_top1: { mean_diff: number; ci95: [number, number] };
}

export interface EvalCompare {
  runs: EvalCompareRun[];
  paired: EvalComparePaired[];
}
