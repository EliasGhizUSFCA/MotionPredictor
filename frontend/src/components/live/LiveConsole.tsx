"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BirdsEye, KinematicsCharts, ProbabilityPanel } from "@/components/live/Analytics";
import CalibrationWizard from "@/components/live/CalibrationWizard";
import Diagnostics from "@/components/live/Diagnostics";
import VideoStage, { type Layers } from "@/components/live/VideoStage";
import { download, getJSON, postJSON } from "@/lib/api";
import { DEFAULT_CONFIG, type CalibrationRequest, type Health, type PipelineConfig, type TrackOut } from "@/lib/types";
import { DEFAULT_STREAM_OPTS, useStream, type StreamOptions } from "@/lib/useStream";
import { fmt } from "@/lib/viz";

const HORIZONS = [0.5, 1, 2, 3, 5];
const CAL_KEY = "motionpredict.calibration";
// Backend Session warning (contract C4) when frame timestamps go backwards; per-person state is reset.
const TS_RESET_PREFIX = "Timestamps went backwards";

function Metric({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="min-w-[88px] rounded border border-line bg-panel px-2 py-1">
      <div className="text-[10px] uppercase tracking-wide text-dim">{label}</div>
      <div className={`font-mono text-sm ${warn ? "text-warn" : "text-ink"}`}>{value}</div>
    </div>
  );
}

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-panel p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-dim">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export default function LiveConsole({ health }: { health: Health | null }) {
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_CONFIG);
  const [opts, setOpts] = useState<StreamOptions>(DEFAULT_STREAM_OPTS);
  const [layers, setLayers] = useState<Layers>({ pose: true, paths: true, cones: true, trails: true, truth: true });
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [wizard, setWizard] = useState<string | null>(null);
  const [calib, setCalib] = useState<CalibrationRequest | null>(null);
  const [scene, setScene] = useState("zara1");
  const [rtsp, setRtsp] = useState("");
  const [renderMs, setRenderMs] = useState<number | null>(null);
  const [explain, setExplain] = useState<Record<string, any> | null>(null);
  const [useLlm, setUseLlm] = useState(false);
  const [consent, setConsent] = useState(false);
  const s = useStream(config, opts);
  const fileRef = useRef<HTMLInputElement>(null);
  const renderAcc = useRef(0);

  useEffect(() => {
    try {
      const c = localStorage.getItem(CAL_KEY);
      if (c) setCalib(JSON.parse(c));
    } catch {}
  }, []);
  useEffect(() => {
    if (s.status === "streaming" && calib && s.source !== "demo") s.setCalibration(calib);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.status === "streaming"]);

  const onRenderMs = useCallback((ms: number) => {
    renderAcc.current = renderAcc.current * 0.9 + ms * 0.1;
  }, []);
  useEffect(() => {
    const id = setInterval(() => setRenderMs(renderAcc.current), 500);
    return () => clearInterval(id);
  }, []);

  const r = s.result;
  // The reset warning only appears on the one frame where it happened; keep it visible for a few seconds.
  const [resetNotice, setResetNotice] = useState<string | null>(null);
  useEffect(() => {
    const w = r?.warnings.find((x) => x.startsWith(TS_RESET_PREFIX));
    if (!w) return;
    setResetNotice(w);
    setSelected(null);
    const id = setTimeout(() => setResetNotice(null), 5000);
    return () => clearTimeout(id);
  }, [r]);
  const track: TrackOut | null = useMemo(() => {
    if (!r) return null;
    const conf = r.tracks.filter((t) => t.state === "confirmed");
    return r.tracks.find((t) => t.track_id === selected) ?? (selected === null ? conf[0] ?? null : null);
  }, [r, selected]);
  const activeId = track?.track_id ?? null;

  const upd = (p: Partial<PipelineConfig>) => setConfig((c) => ({ ...c, ...p }));
  const applyCal = (c: CalibrationRequest | null) => {
    setCalib(c);
    try {
      if (c) localStorage.setItem(CAL_KEY, JSON.stringify(c));
      else localStorage.removeItem(CAL_KEY);
    } catch {}
    s.setCalibration(c);
  };
  const openWizard = () => {
    const v = s.videoRef.current;
    if (s.demoFrame) return setWizard(s.demoFrame);
    if (!v || !v.videoWidth) return setWizard("");
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    setWizard(c.toDataURL("image/jpeg", 0.85)); // kept in memory only; never uploaded
  };
  const exportSession = () => {
    // non-sensitive: kinematics + trajectories + timings; no images, no keypoints
    const frames = s.history.current.map((f) => ({
      frame_id: f.frame_id, t_ms: f.server_done_ms, processing_ms: f.processing_ms, stats: f.stats,
      stage_ms: Object.fromEntries(f.events.map((e) => [e.stage, e.duration_ms])),
      tracks: f.tracks.map((t) => ({
        track_id: t.track_id, state: t.state, footpoint_px: t.footpoint_px, kinematics: t.kinematics, kinematics_world: t.kinematics_world,
        height: t.height, forecast: t.forecast && { ...t.forecast, trajectories: t.forecast.trajectories.map(({ band_left, band_right, ...rest }) => rest) },
      })),
    }));
    download(`motionpredict-session-${Date.now()}.json`, JSON.stringify({ exported: new Date().toISOString(), config, calibration: r?.camera_calibration, frames }, null, 1));
  };
  const doExplain = async () => {
    if (!r) return;
    try {
      setExplain(
        await postJSON("/api/explain", {
          track_id: activeId,
          use_llm: useLlm,
          result: r.source === "dataset_replay" ? { ...r, held_out_for_active_model: s.demoInfo?.held_out_for_active_model ?? null } : r,
        }),
      );
    } catch (e: any) {
      setExplain({ ok: false, error: String(e.message ?? e) });
    }
  };

  const running = s.status === "streaming" || s.status === "paused" || s.status === "connecting";
  const metric = r?.camera_calibration.coordinate_system === "ground_plane_meters";
  // Incoming frame rate: the browser measures its own <video> (webcam/file); server-side and
  // dataset-replay sources only have the backend's stats.input_fps.
  const browserSource = s.source === "webcam" || s.source === "file";
  const inputFps = browserSource ? s.metrics.cameraFps ?? r?.stats.input_fps ?? null : r?.stats.input_fps ?? null;
  const inputFpsLabel = browserSource && s.metrics.cameraFps !== null && s.metrics.cameraFps !== undefined ? "camera fps" : "input fps";
  const computeMs = r?.stats.compute_ms_mean ?? null;
  const queueMs = r?.stats.queue_ms_mean ?? null;

  return (
    <div className="space-y-3">
      {/* performance header */}
      <div className="flex flex-wrap gap-2">
        <Metric label="status" value={s.status} warn={s.status === "error"} />
        <Metric label={inputFpsLabel} value={fmt(inputFps, 1)} />
        <Metric label="results fps" value={fmt(s.metrics.resultFps, 1)} />
        <Metric label="inference fps" value={fmt(r?.stats.inference_fps, 1)} />
        <Metric label="e2e latency" value={s.metrics.latencyMs ? `${s.metrics.latencyMs.toFixed(0)} ms` : "–"} warn={(s.metrics.latencyMs ?? 0) > 500} />
        <Metric label="server proc (e2e)" value={r ? `${r.processing_ms.toFixed(0)} ms` : "–"} />
        <Metric label="compute (mean)" value={computeMs !== null ? `${computeMs.toFixed(0)} ms` : "–"} />
        <Metric label="queue wait (mean)" value={queueMs !== null ? `${queueMs.toFixed(0)} ms` : "–"} warn={queueMs !== null && computeMs !== null && queueMs > computeMs} />
        <Metric label="resolution" value={s.metrics.resolution} />
        <Metric label="skipped busy / server drop" value={`${s.metrics.clientDropped}/${s.metrics.serverDropped}`} />
        <Metric label="device" value={health?.device.device ?? "–"} />
        <Metric label="forecaster" value={r?.stats.forecaster_active === "trained_model" ? "learned" : r?.stats.forecaster_active === "physics_baseline" ? "baseline" : "–"} />
        <Metric label="calibration" value={r?.camera_calibration.status ?? (calib ? "pending" : "none")} warn={!metric} />
      </div>
      {resetNotice && <div className="rounded border border-warn/50 bg-warn/10 px-3 py-2 text-sm text-warn">⚠ {resetNotice}</div>}
      {s.error && <div className="rounded border border-bad/50 bg-bad/10 px-3 py-2 text-sm text-bad">{s.error}</div>}
      {s.source === "demo" && (
        <div className="rounded border border-accent/40 bg-accent/10 px-3 py-2 text-xs text-ink">
          DEMO MODE · dataset replay of human-annotated ETH/UCY positions ({s.demoInfo?.file}). The detector is bypassed and the image is a synthetic top-down rendering.
          {s.demoInfo?.held_out_for_active_model === true
            ? " This scene was held out from the active model's training."
            : s.demoInfo?.held_out_for_active_model === false
              ? " Note: the active model was trained on this scene."
              : " Whether the active model was trained on this scene is unknown."}
          {" "}White dotted lines show what actually happened next.
        </div>
      )}

      <div className="grid gap-3 xl:grid-cols-[290px_minmax(0,1fr)_340px]">
        {/* left: controls + tracks */}
        <div className="space-y-3">
          <Section title="Source">
            {!consent && (
              <div className="mb-2 rounded border border-line bg-panel2 p-2 text-[11px] text-dim">
                Camera frames are processed by your local backend only. Nothing is recorded or stored; track IDs are temporary and cleared when you stop.
                No face recognition or identity inference is performed.
                <button className="btn btn-primary mt-2 w-full" onClick={() => setConsent(true)}>I understand</button>
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <button className="btn btn-primary" disabled={!consent || running} onClick={() => s.start("webcam")}>Start camera</button>
              <button className="btn" disabled={!running} onClick={() => { s.stop(); setSelected(null); }}>Stop</button>
              <button className="btn" disabled={!running} onClick={() => s.setPaused(s.status !== "paused")}>{s.status === "paused" ? "Resume" : "Pause"}</button>
              <button className="btn" disabled={!running} onClick={s.reset}>Reset tracks</button>
              <button className="btn col-span-2" disabled={!consent} onClick={() => fileRef.current?.click()}>Open video file…</button>
              <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && s.start("file", e.target.files[0])} />
            </div>
            <div className="mt-2 flex gap-2">
              <select className="inp flex-1" value={scene} onChange={(e) => setScene(e.target.value)}>
                {["zara1", "zara2", "eth", "hotel", "univ"].map((x) => <option key={x}>{x}</option>)}
              </select>
              <button className="btn" disabled={!health?.demo_available} onClick={() => s.start("demo", scene)} title={health?.demo_available ? "" : "run scripts/download_data.sh"}>Demo replay</button>
            </div>
            {health?.server_sources_allowed && (
              <div className="mt-2 flex gap-2">
                <input className="inp flex-1" placeholder="rtsp://… (local, authorised)" value={rtsp} onChange={(e) => setRtsp(e.target.value)} />
                <button className="btn" disabled={!rtsp} onClick={() => s.start("server", rtsp)}>Connect</button>
              </div>
            )}
            <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-dim">
              <label>camera res{s.source === "webcam" && running ? <span className="text-warn"> · applies on next start</span> : ""}
                <select className="inp w-full" value={`${opts.camWidth}x${opts.camHeight}`} onChange={(e) => { const [w, h] = e.target.value.split("x").map(Number); setOpts({ ...opts, camWidth: w, camHeight: h }); }}>
                  {["640x480", "1280x720", "1920x1080"].map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <label>inference width
                <select className="inp w-full" value={opts.sendWidth} onChange={(e) => setOpts({ ...opts, sendWidth: Number(e.target.value) })}>
                  {[320, 480, 640, 960].map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <label>max send fps
                <select className="inp w-full" value={opts.maxSendFps} onChange={(e) => setOpts({ ...opts, maxSendFps: Number(e.target.value) })}>
                  {[5, 10, 15, 20, 30].map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <label>detector size
                <select className="inp w-full" value={config.imgsz} onChange={(e) => upd({ imgsz: Number(e.target.value) })}>
                  {[320, 480, 640, 960].map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
            </div>
          </Section>

          <Section title="Forecast">
            <div className="mb-2 grid grid-cols-2 gap-1">
              {(["learned", "baseline"] as const).map((m) => (
                <button key={m} className={`tab ${config.forecaster === m ? "tab-on" : ""}`} onClick={() => upd({ forecaster: m })}>
                  {m === "learned" ? "Learned (SocialMDN)" : "Physics baseline"}
                </button>
              ))}
            </div>
            <div className="mb-2 flex gap-1">
              {HORIZONS.map((h) => (
                <button key={h} className={`tab flex-1 ${config.horizon_s === h ? "tab-on" : ""}`} onClick={() => upd({ horizon_s: h })}>{h}s</button>
              ))}
            </div>
            <label className="block text-[11px] text-dim">hypotheses: {config.num_hypotheses}
              <input type="range" min={1} max={5} value={config.num_hypotheses} onChange={(e) => upd({ num_hypotheses: Number(e.target.value) })} className="w-full" />
            </label>
            <label className="block text-[11px] text-dim">detector confidence: {config.det_conf.toFixed(2)}
              <input type="range" min={0.1} max={0.9} step={0.05} value={config.det_conf} onChange={(e) => upd({ det_conf: Number(e.target.value) })} className="w-full" />
            </label>
            <label className="block text-[11px] text-dim">NMS IoU: {config.det_iou.toFixed(2)}
              <input type="range" min={0.2} max={0.9} step={0.05} value={config.det_iou} onChange={(e) => upd({ det_iou: Number(e.target.value) })} className="w-full" />
            </label>
            <label className="block text-[11px] text-dim">pose every {config.pose_every} frame(s) - lower is smoother, higher is faster
              <input type="range" min={1} max={10} value={config.pose_every} onChange={(e) => upd({ pose_every: Number(e.target.value) })} className="w-full" />
            </label>
            <label className="block text-[11px] text-dim">track history: {config.history_s}s
              <input type="range" min={2} max={20} value={config.history_s} onChange={(e) => upd({ history_s: Number(e.target.value) })} className="w-full" />
            </label>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
              {(Object.keys(layers) as (keyof Layers)[]).map((k) => (
                <label key={k} className="flex items-center gap-1">
                  <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />
                  {{ pose: "skeleton overlay", paths: "paths", cones: "uncertainty", trails: "trails", truth: "true future (replay)" }[k]}
                </label>
              ))}
            </div>
            {/* Pose ESTIMATION (backend) is separate from skeleton DISPLAY (overlay layer above). */}
            <label className="mt-1 flex items-center gap-1 text-xs" title="Runs the pose model on the backend. Needed for the upright/full-body check behind metric height; hiding the skeleton overlay does not turn it off.">
              <input type="checkbox" checked={config.pose} onChange={(e) => upd({ pose: e.target.checked })} />
              pose estimation
            </label>
            {!config.pose && <p className="text-[11px] text-warn">Pose estimation off: no skeletons, and metric height is unavailable (it needs pose keypoints).</p>}
            <div className="mt-2 grid grid-cols-2 gap-2">
              <button className="btn" onClick={openWizard}>Calibrate…</button>
              <button className="btn" disabled={!r} onClick={exportSession}>Export session</button>
            </div>
          </Section>

          <Section title={`Tracks (${r?.tracks.length ?? 0})`}>
            <div className="max-h-64 space-y-1 overflow-auto">
              {r?.tracks.map((t) => {
                const k = t.kinematics_world ?? t.kinematics;
                return (
                  <button key={t.track_id} onClick={() => setSelected(t.track_id)} className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs ${t.track_id === activeId ? "bg-accent/15 text-ink" : "text-dim hover:bg-panel2"}`}>
                    <span className="font-mono text-ink">#{t.track_id}</span>
                    <span className={t.state === "confirmed" ? "text-good" : t.state === "lost" ? "text-warn" : "text-dim"}>{t.state}</span>
                    <span className="ml-auto font-mono">{k ? `${k.speed.toFixed(k.units === "m" ? 2 : 0)} ${k.units}/s` : ""}</span>
                    <span className="font-mono">{(t.tracker_confidence * 100).toFixed(0)}%</span>
                  </button>
                );
              })}
              {!r?.tracks.length && <p className="text-xs text-dim">No people tracked.</p>}
            </div>
            {track && (
              <div className="mt-2 space-y-0.5 border-t border-line pt-2 text-[11px] text-dim">
                <p className="text-ink">Track #{track.track_id} · {track.state} · age {track.age_s.toFixed(1)}s · {track.hits} hits</p>
                <p>tracker conf {fmt(track.tracker_confidence)} · detection {fmt(track.det_confidence)}</p>
                {(() => {
                  const k = track.kinematics_world ?? track.kinematics;
                  if (!k) return <p>kinematics: warming up</p>;
                  const u = k.units;
                  return (
                    <>
                      <p>speed {fmt(k.speed)} {u}/s {k.speed_category ? `(${k.speed_category})` : ""} · {k.movement_state}</p>
                      <p>heading {fmt(k.heading_deg, 0)}° · turn {fmt(k.turn_rate_deg_s, 0)}°/s · accel {fmt(k.accel_magnitude)} {u}/s²</p>
                      <p>velocity ({fmt(k.velocity[0])}, {fmt(k.velocity[1])}) {u}/s · pos ±{fmt(k.position_std)} {u}</p>
                    </>
                  );
                })()}
                <p>
                  height:{" "}
                  {track.height.status === "calibrated"
                    ? `${fmt(track.height.meters)} m (≈${fmt(track.height.interval_m?.[0])}–${fmt(track.height.interval_m?.[1])} m)`
                    : `unknown - ${track.height.reason}`}
                  {track.height.relative_image_height !== null ? ` · ${(track.height.relative_image_height * 100).toFixed(0)}% of frame height` : ""}
                </p>
                <p className="text-[10px]">Momentum is not computed: it would need the person&apos;s mass.</p>
              </div>
            )}
          </Section>
        </div>

        {/* centre: video + map + charts */}
        <div className="min-w-0 space-y-3">
          <VideoStage videoRef={s.videoRef} result={r} demoFrame={s.source === "demo" || s.source === "server" ? s.demoFrame : null} layers={layers}
            selected={activeId} onSelect={setSelected} showVideo={s.source === "webcam" || s.source === "file"} onRenderMs={onRenderMs} />
          <div className="flex flex-wrap gap-3 text-[11px] text-dim">
            <span><span className="text-obs">━</span> observed history / velocity</span>
            <span><span className="text-good">┅</span> most likely</span>
            <span><span className="text-warn">┅</span> next</span>
            <span><span className="text-bad">┅</span> lower</span>
            <span>shaded = 2σ cross-track uncertainty · white arrow = heading</span>
          </div>
          <div className="grid gap-3 lg:grid-cols-2">
            <Section title="Bird's-eye trajectory map"><BirdsEye result={r} selected={activeId} onSelect={setSelected} highlight={highlight} setHighlight={setHighlight} /></Section>
            <Section title="Path hypotheses"><ProbabilityPanel track={track} setHighlight={setHighlight} /></Section>
          </div>
          <Section title="Motion time series"><KinematicsCharts history={s.history.current} trackId={activeId} /></Section>
        </div>

        {/* right: diagnostics + explanation */}
        <div className="space-y-3">
          <Section title="AI pipeline diagnostics"><Diagnostics result={r} renderMs={renderMs} /></Section>
          <Section title="Explain">
            <p className="mb-2 text-[11px] text-dim">Summarises the computed numbers for the selected track. The LLM only rephrases them; it never detects or predicts.</p>
            <label className={`mb-2 flex items-center gap-2 text-xs ${health?.llm.configured ? "" : "opacity-50"}`}>
              <input type="checkbox" disabled={!health?.llm.configured} checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
              also ask Claude ({health?.llm.model}) {health?.llm.configured ? "- sends structured numbers, no images" : "- set ANTHROPIC_API_KEY on the backend"}
            </label>
            <button className="btn w-full" disabled={!r} onClick={doExplain}>Explain track {activeId ?? ""}</button>
            {explain && (
              <div className="mt-2 space-y-2 text-xs">
                {explain.ok === false && <p className="text-bad">{explain.error}</p>}
                {explain.rule_based && <div><p className="text-[10px] uppercase text-dim">rule-based summary</p><p className="text-ink">{explain.rule_based}</p></div>}
                {explain.llm && (
                  <div>
                    <p className="text-[10px] uppercase text-dim">LLM summary ({explain.llm.model ?? "Claude"})</p>
                    <p className={explain.llm.ok ? "text-ink" : "text-bad"}>{explain.llm.ok ? explain.llm.text : explain.llm.error}</p>
                  </div>
                )}
              </div>
            )}
          </Section>
        </div>
      </div>
      {wizard !== null && <CalibrationWizard snapshot={wizard || null} current={calib} onApply={applyCal} onClose={() => setWizard(null)} />}
    </div>
  );
}
