"use client";
// Forecast replay: shows the forecast exactly as stored at prediction time, then reveals what
// actually happened. Nothing is re-predicted here - the stored record is the source of truth.

import { useEffect, useMemo, useRef, useState } from "react";
import TrajectoryMap, { type MapItem } from "@/components/ui/TrajectoryMap";
import { postJSON } from "@/lib/api";
import type { EvalPrediction, Pt } from "@/lib/types";
import { ACTUAL_COLOR, GT_END_TOLERANCE_S, applyH, hk, band, bandFromSigma, fmt, interpAt, inv3, nearestForecast, partialErrors, polyline, rankColor, rgba, trackPosAt, valid } from "@/lib/viz";
import type { RunData } from "./EvalTab";

const MODEL_LABEL: Record<string, string> = { baseline: "A · Kalman baseline", learned: "B · SocialMDN" };
const MODEL_SHORT: Record<string, string> = { baseline: "A", learned: "B" };
const CORR_COLOR = "#f97316";
// bare percentages are only allowed for validated probabilities
const pctLabel = (prob: number, type: string) => `${(prob * 100).toFixed(0)}%${type === "calibrated_probability" ? "" : " (uncal.)"}`;

export default function ReplayPanel({ data, videoUrl, focus, onFlagged }: {
  data: RunData;
  videoUrl: string | null;
  focus: { id: string; h?: number } | null;
  onFlagged: () => void;
}) {
  const { predictions, summary, tracks, info } = data;
  const meta = info.meta!;
  const metric = summary.coordinate_system === "ground_plane_meters";
  const W: number = meta.video?.width ?? 1280;
  const H: number = meta.video?.height ?? 720;
  const Hinv: number[][] | undefined = meta.calibration?.homography_inv;
  const [pid, setPid] = useState<string>(focus?.id ?? summary.failure_cases[0]?.prediction_id ?? predictions[0]?.id);
  const [h, setH] = useState<number>(focus?.h ?? summary.selected_horizon_for_rankings);
  const [shown, setShown] = useState<string[]>(summary.models);
  const [reveal, setReveal] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [note, setNote] = useState("");
  const [endedAt, setEndedAt] = useState<number | null>(null); // reveal (s) at which the recording ended before t + h
  const [browseT, setBrowseT] = useState<number | null>(null); // free video time while browsing (no forecast shown)
  const [correcting, setCorrecting] = useState(false);
  const [corr, setCorr] = useState<{ t: number; px: Pt; c: Pt }[]>([]); // corrected actual future: t_rel, frame px, run coords
  const [msg, setMsg] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (focus) {
      setBrowseT(null);
      setPid(focus.id);
      if (focus.h) setH(focus.h);
    }
  }, [focus]);

  const p: EvalPrediction | undefined = useMemo(() => predictions.find((x) => x.id === pid), [predictions, pid]);
  const trackPreds = useMemo(() => (p ? predictions.filter((x) => x.track_id === p.track_id) : []), [predictions, p]);
  const trackIds = useMemo(() => Array.from(new Set(predictions.map((x) => x.track_id))).sort((a, b) => a - b), [predictions]);
  const obs = useMemo(() => (p ? tracks[String(p.track_id)] ?? [] : []), [tracks, p]);
  const tMax = useMemo(() => {
    let m = Number(meta.video?.duration_s) || 0;
    for (const o of Object.values(tracks)) if (o.length) m = Math.max(m, o[o.length - 1][0]);
    for (const x of predictions) m = Math.max(m, x.t);
    return m;
  }, [tracks, predictions, meta.video?.duration_s]);
  // image -> run coordinates (metric runs: calibrated homography maps normalised image coords to the ground plane)
  const Hfwd: number[][] | null = meta.calibration?.homography ?? (Hinv ? inv3(Hinv) : null);
  const toCoord = (q: Pt): Pt | null => {
    if (!metric) return q;
    if (!Hfwd || !p) return null;
    const sign = Math.sign(Hfwd[2][0] * (p.start_px[0] / W) + Hfwd[2][1] * (p.start_px[1] / H) + Hfwd[2][2]) || 1;
    return applyH(Hfwd, [q[0] / W, q[1] / H], sign);
  };

  // keyed on the prediction id (not the object) so an in-place data refresh after flagging keeps the replay state
  useEffect(() => {
    setReveal(0);
    setPlaying(false);
    setEndedAt(null);
    setCorr([]);
    setCorrecting(false);
    setMsg(null);
    const v = videoRef.current;
    if (v && p) {
      v.pause();
      v.currentTime = p.t;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid, p?.id]);

  useEffect(() => {
    const v = videoRef.current;
    if (v && browseT !== null) {
      v.pause();
      v.currentTime = browseT;
    }
  }, [browseT]);

  // play-through: video time drives the reveal (or a timer when no video is attached)
  useEffect(() => {
    if (!playing || !p) return;
    const v = videoRef.current;
    let raf = 0;
    const t0 = performance.now() - reveal * 1000;
    if (v && videoUrl) {
      v.currentTime = p.t + reveal;
      v.play()?.catch(() => {});
    }
    const tick = () => {
      const vid = !!(v && videoUrl);
      const r = vid ? v!.currentTime - p.t : (performance.now() - t0) / 1000;
      // the recording can end before t + h (forecasts near the end are kept and flagged 'recording_ended')
      const atEnd = vid && (v!.ended || (Number.isFinite(v!.duration) && v!.currentTime >= v!.duration - 0.05));
      if (r >= h || atEnd) {
        const short = r < h && h - r > GT_END_TOLERANCE_S;
        setReveal(short ? Math.max(0, r) : h);
        setEndedAt(short ? Math.max(0, r) : null);
        setPlaying(false);
        v?.pause();
        return;
      }
      setReveal(Math.max(0, r));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      v?.pause();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  const scrub = (r: number) => {
    setPlaying(false);
    setBrowseT(null);
    setEndedAt(null);
    setReveal(r);
    if (videoRef.current && p) videoRef.current.currentTime = p.t + r;
  };

  // coordinate-system path -> frame pixels
  const toPx = (q: Pt): Pt | null => {
    if (!metric) return q;
    if (!Hinv) return null;
    const sign = Math.sign(Hinv[2][0] * (p?.start[0] ?? 0) + Hinv[2][1] * (p?.start[1] ?? 0) + Hinv[2][2]) || 1;
    const n = applyH(Hinv, q, sign);
    return n ? [n[0] * W, n[1] * H] : null;
  };

  // overlay drawing
  useEffect(() => {
    const c = canvas.current;
    if (!c || !p) return;
    const rect = c.getBoundingClientRect();
    c.width = rect.width * devicePixelRatio;
    c.height = rect.height * devicePixelRatio;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(c.width / W, 0, 0, c.height / H, 0, 0);
    const u = W / rect.width;
    ctx.clearRect(0, 0, W, H);
    if (!videoUrl) {
      ctx.fillStyle = "#0b0f14";
      ctx.fillRect(0, 0, W, H);
    }
    // everyone observed at the current time (click one to select their track)
    const now = browseT ?? p.t + reveal;
    ctx.font = `${10 * u}px ui-monospace, monospace`;
    for (const [tid, o] of Object.entries(tracks)) {
      const q = trackPosAt(o, now);
      if (!q) continue;
      const sel = Number(tid) === p.track_id;
      ctx.strokeStyle = sel ? "#ffffff" : "rgba(226,232,240,0.55)";
      ctx.lineWidth = (sel ? 2 : 1.2) * u;
      ctx.beginPath();
      ctx.arc(q[0], q[1], 7 * u, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = sel ? "#ffffff" : "rgba(226,232,240,0.7)";
      ctx.fillText(`#${tid}`, q[0] + 8 * u, q[1] + 3 * u);
    }
    if (browseT !== null) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.font = `${12 * devicePixelRatio}px ui-monospace, monospace`;
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(`t = ${browseT.toFixed(2)} s  (browsing - click a person to open their nearest stored forecast)`, 8 * devicePixelRatio, c.height - 8 * devicePixelRatio);
      return;
    }
    // observed history before t (solid blue)
    const hist = obs.filter((o) => o[0] <= p.t + 1e-6 && o[0] >= p.t - 6).map((o) => [o[1], o[2]] as Pt);
    ctx.strokeStyle = rgba(ACTUAL_COLOR, 0.55);
    ctx.lineWidth = 2 * u;
    polyline(ctx, hist);
    // predictions (dashed), per model
    for (const m of shown) {
      const mo = p.models[m];
      if (!mo) continue;
      const trajs = mo.trajectories.filter((t) => m !== "learned" || t.rank < 3);
      for (const t of [...trajs].sort((a, b) => b.rank - a.rank)) {
        const keep = t.times.map((x) => x <= h + 1e-6);
        const pts = t.points.filter((_, i) => keep[i]);
        const sig = t.sigma.filter((_, i) => keep[i]);
        const col = rankColor(t.rank);
        if (sig.some((s) => s[0] > 0)) {
          const b = bandFromSigma(pts, sig, 2);
          ctx.fillStyle = rgba(col, 0.1);
          band(ctx, b.left.map(toPx).map((x) => x ?? [null, null]) as any, b.right.map(toPx).map((x) => x ?? [null, null]) as any);
        }
        ctx.strokeStyle = col;
        ctx.lineWidth = (t.rank === 0 ? 2.5 : 1.8) * u;
        ctx.setLineDash(m === "baseline" ? [3 * u, 4 * u] : [9 * u, 5 * u]);
        polyline(ctx, pts.map((q) => toPx(q) ?? ([null, null] as any)));
        ctx.setLineDash([]);
        const at = interpAt(t.times, t.points, reveal);
        const atPx = at && toPx(at);
        if (atPx && reveal > 0) {
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(atPx[0], atPx[1], 3 * u, 0, Math.PI * 2);
          ctx.fill();
        }
        const end = toPx(pts[pts.length - 1]);
        if (end) {
          ctx.fillStyle = col;
          ctx.font = `${11 * u}px ui-monospace, monospace`;
          ctx.fillText(`${MODEL_SHORT[m]}:${t.id}${t.probability !== null ? ` ${pctLabel(t.probability, mo.probability_type)}` : ""}`, end[0] + 4 * u, end[1] - 4 * u);
        }
      }
    }
    // actual future revealed so far (solid blue)
    const fut = obs.filter((o) => o[0] >= p.t - 1e-6 && o[0] <= p.t + reveal + 1e-6).map((o) => [o[1], o[2]] as Pt);
    ctx.strokeStyle = ACTUAL_COLOR;
    ctx.lineWidth = 3.5 * u;
    polyline(ctx, fut);
    const cur = fut.at(-1);
    if (cur && reveal > 0) {
      ctx.fillStyle = ACTUAL_COLOR;
      ctx.beginPath();
      ctx.arc(cur[0], cur[1], 5 * u, 0, Math.PI * 2);
      ctx.fill();
    }
    // user-corrected actual future (orange)
    if (corr.length) {
      ctx.strokeStyle = CORR_COLOR;
      ctx.fillStyle = CORR_COLOR;
      ctx.lineWidth = 2.5 * u;
      polyline(ctx, [p.start_px, ...corr.map((x) => x.px)]);
      for (const x of corr) {
        ctx.beginPath();
        ctx.arc(x.px[0], x.px[1], 4 * u, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // start marker
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.5 * u;
    ctx.beginPath();
    ctx.arc(p.start_px[0], p.start_px[1], 5.5 * u, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = `${12 * devicePixelRatio}px ui-monospace, monospace`;
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(`t = ${(p.t + reveal).toFixed(2)} s  (forecast made at ${p.t.toFixed(2)} s, +${reveal.toFixed(2)} s revealed${endedAt !== null ? ", recording ended" : ""})`, 8 * devicePixelRatio, c.height - 8 * devicePixelRatio);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p, shown, reveal, h, obs, videoUrl, W, H, browseT, tracks, corr, endedAt]);

  if (!p) return <div className="text-sm text-dim">No predictions in this run.</div>;

  const act = p.actual_future.map((a) => [a[1], a[2]] as Pt);
  const actT = p.actual_future.map((a) => a[0]);
  const ev = p.evaluations[hk(h)];
  const unit = metric ? "m" : "px";

  const mapItems: MapItem[] = [];
  mapItems.push({ kind: "line", pts: p.observation_window.positions, color: ACTUAL_COLOR, width: 2, alpha: 0.6 });
  for (const m of shown) {
    const mo = p.models[m];
    if (!mo) continue;
    for (const t of mo.trajectories.filter((x) => m !== "learned" || x.rank < 3)) {
      const keep = t.times.map((x) => x <= h + 1e-6);
      const pts = t.points.filter((_, i) => keep[i]);
      const col = rankColor(t.rank);
      const i = t.times.findIndex((x) => x >= reveal - 1e-6);
      if (i > 0 && t.sigma[i][0] > 0 && keep[i]) mapItems.push({ kind: "ellipse", at: t.points[i], ellipse: t.sigma[i], color: col, width: 1, alpha: 0.6 });
      mapItems.push({ kind: "line", pts, color: col, width: t.rank === 0 ? 2.5 : 1.6, dash: m === "baseline" ? [2, 4] : [7, 4] });
    }
  }
  const revealed = act.filter((_, i) => actT[i] <= reveal + 1e-6);
  if (browseT === null && revealed.length > 1) mapItems.push({ kind: "line", pts: revealed, color: ACTUAL_COLOR, width: 3.5 });
  if (corr.length) mapItems.push({ kind: "line", pts: [p.start, ...corr.map((x) => x.c)], color: CORR_COLOR, width: 2.5 });
  mapItems.push({ kind: "point", at: p.start, color: "#ffffff", radius: 5 });
  // everyone observed at the current time, clickable (id = track id)
  const now = browseT ?? p.t + reveal;
  for (const [tid, o] of Object.entries(tracks)) {
    const rows = metric ? o.filter((r) => r[3] !== null && r[4] !== null).map((r) => [r[0], r[3]!, r[4]!]) : o;
    const q = trackPosAt(rows, now);
    if (q) mapItems.push({ kind: "point", at: q, pts: [q], id: tid, color: Number(tid) === p.track_id ? "#ffffff" : "#94a3b8", radius: 3 });
  }

  const flag = async (kind: string) => {
    try {
      await postJSON(`/api/eval/runs/${summary.run_id}/flags`, { kind, prediction_id: kind === "hard_case" ? p.id : null, track_id: p.track_id, note });
      setNote("");
      setMsg(kind === "hard_case" ? "flagged as hard case" : `track #${p.track_id} flagged as inaccurate - excluded from all statistics`);
      onFlagged();
    } catch (e: any) {
      setMsg(String(e.message ?? e));
    }
  };
  const unflagTrack = async () => {
    await postJSON(`/api/eval/runs/${summary.run_id}/flags`, { kind: "unflag", prediction_id: null, track_id: p.track_id });
    setMsg(`track #${p.track_id} restored to the statistics`);
    onFlagged();
  };
  const trackFlagged = summary.flagged_tracks_excluded.includes(p.track_id);

  // open the stored forecast for a track nearest to time t (latest one made at or before t)
  const openTrackAt = (track: number, t: number) => {
    const f = nearestForecast(predictions.filter((x) => x.track_id === track), t);
    if (!f) {
      setMsg(`no stored forecast for track #${track}`);
      return;
    }
    if (f.id === pid) scrub(0); // same forecast: just return the video to its forecast time
    setBrowseT(null);
    setPid(f.id);
    setMsg(Math.abs(f.t - t) > 0.05 ? `track #${track}: nearest stored forecast is at ${f.t.toFixed(2)} s (forecasts are stored every ${meta.eval_config?.interval_s ?? "?"} s)` : null);
  };

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const q: Pt = [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
    if (correcting) {
      const tRel = now - p.t;
      if (tRel <= 1e-3 || tRel > p.hmax + 1e-6) {
        setMsg(`move the reveal slider to a time between +0 and +${p.hmax} s, then click where the person actually is`);
        return;
      }
      const c = toCoord(q);
      if (!c) {
        setMsg("cannot convert this point to ground-plane coordinates (no usable calibration)");
        return;
      }
      setCorr((xs) => [...xs.filter((x) => Math.abs(x.t - tRel) > 0.02), { t: Math.round(tRel * 1000) / 1000, px: q, c }].sort((a, b) => a.t - b.t));
      return;
    }
    // pick the nearest person observed at this time
    let best: { id: number; d: number } | null = null;
    for (const [tid, o] of Object.entries(tracks)) {
      const pos = trackPosAt(o, now);
      if (!pos) continue;
      const d = Math.hypot(pos[0] - q[0], pos[1] - q[1]);
      if (d < (30 * W) / r.width && (!best || d < best.d)) best = { id: Number(tid), d };
    }
    if (best) openTrackAt(best.id, now);
  };

  const saveCorrection = async () => {
    const pts: [number, number, number][] = corr.map((x) => [x.t, x.c[0], x.c[1]]);
    if (!pts.length || pts[0][0] > 0) pts.unshift([0, p.start[0], p.start[1]]); // starts at the observed position at forecast time
    try {
      await postJSON(`/api/eval/runs/${summary.run_id}/flags`, { kind: "annotation_correction", prediction_id: p.id, track_id: p.track_id, note, corrected_future: pts });
      setNote("");
      setCorrecting(false);
      setMsg(`correction saved (${pts.length} points, ${metric ? "m" : "px"})`);
      onFlagged();
    } catch (e: any) {
      setMsg(String(e.message ?? e));
    }
  };
  const idx = trackPreds.findIndex((x) => x.id === p.id);

  return (
    <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-2">
        <div className="relative w-full overflow-hidden rounded-lg border border-line bg-black" style={{ aspectRatio: `${W}/${H}` }}>
          {videoUrl && <video ref={videoRef} src={videoUrl} className="absolute inset-0 h-full w-full object-fill" muted playsInline preload="auto" />}
          <canvas ref={canvas} className={`absolute inset-0 h-full w-full ${correcting ? "cursor-crosshair" : "cursor-pointer"}`} onClick={onCanvasClick} />
          {!videoUrl && meta.source?.type === "video" && (
            <div className="absolute right-2 top-2 rounded bg-panel/80 px-2 py-1 text-[11px] text-dim">Attach the local video file above to see the footage under the overlay.</div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button className="btn btn-primary" onClick={() => { if (reveal >= h || endedAt !== null || browseT !== null) scrub(0); setPlaying(true); }} disabled={playing}>
            {reveal > 0 && reveal < h && endedAt === null && browseT === null ? "Continue" : "Reveal future"} ▶
          </button>
          <button className="btn" onClick={() => setPlaying(false)} disabled={!playing}>Pause</button>
          <button className="btn" onClick={() => scrub(0)}>Back to forecast time</button>
          <input type="range" min={0} max={h} step={0.05} value={reveal} onChange={(e) => scrub(Number(e.target.value))} className="flex-1" aria-label="reveal slider" />
          <span className="font-mono text-dim">+{reveal.toFixed(2)} / {h}s</span>
        </div>
        {endedAt !== null && browseT === null && (
          <p className="text-[11px] text-warn">The recording ends at +{endedAt.toFixed(2)} s, before this horizon; there is no actual position at +{h} s.</p>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-[11px] text-dim">video time</span>
          <input type="range" min={0} max={Math.max(tMax, 0.1)} step={0.05} value={Math.min(now, tMax)} onChange={(e) => { setPlaying(false); setBrowseT(Number(e.target.value)); }} className="flex-1" aria-label="video time" />
          <span className="font-mono text-dim">{now.toFixed(2)} s</span>
          {browseT !== null && (
            <>
              <button className="btn" onClick={() => openTrackAt(p.track_id, browseT)}>Nearest forecast for #{p.track_id}</button>
              <button className="btn" onClick={() => scrub(0)}>Back to forecast</button>
            </>
          )}
        </div>
        <p className="text-[11px] text-dim">Drag &ldquo;video time&rdquo; to any moment and click a person (on the video or the map) to open their nearest stored forecast made at or before that time.</p>
        <div className="flex flex-wrap gap-3 text-[11px] text-dim">
          <span><span className="text-good">┅</span> most likely</span><span><span className="text-warn">┅</span> secondary</span><span><span className="text-bad">┅</span> lower</span>
          <span><span className="text-obs">━</span> actual (revealed)</span><span>○ start · long dashes = B (learned), dots = A (baseline) · shading = 2σ</span>
        </div>
        <TrajectoryMap items={mapItems} metric={metric} height={260} fitKey={p.id} title={metric ? "Top-down ground plane (m)" : "Image-plane coordinates (px)"} onPick={(id) => !correcting && openTrackAt(Number(id), now)} />
      </div>

      <div className="space-y-3">
        <section className="rounded-lg border border-line bg-panel p-3 text-xs">
          <div className="mb-2 grid grid-cols-2 gap-2">
            <label className="text-[11px] text-dim">track
              <select className="inp w-full" value={p.track_id} onChange={(e) => openTrackAt(Number(e.target.value), browseT ?? p.t)}>
                {trackIds.map((t) => <option key={t} value={t}>#{t}{summary.flagged_tracks_excluded.includes(t) ? " (flagged inaccurate)" : ""}</option>)}
              </select>
            </label>
            <label className="text-[11px] text-dim">horizon
              <select className="inp w-full" value={h} onChange={(e) => { setH(Number(e.target.value)); scrub(0); }}>
                {summary.horizons.map((x) => <option key={x} value={x}>{x}s</option>)}
              </select>
            </label>
          </div>
          <label className="block text-[11px] text-dim">forecast time ({idx + 1}/{trackPreds.length}) · {p.t.toFixed(2)} s
            <input type="range" min={0} max={Math.max(trackPreds.length - 1, 0)} value={Math.max(idx, 0)} onChange={(e) => setPid(trackPreds[Number(e.target.value)].id)} className="w-full" />
          </label>
          <div className="mt-2 flex gap-1">
            {summary.models.map((m) => (
              <button key={m} className={`tab flex-1 ${shown.includes(m) ? "tab-on" : ""}`} onClick={() => setShown((s) => (s.includes(m) ? s.filter((x) => x !== m) : [...s, m]))}>{MODEL_LABEL[m] ?? m}</button>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-line bg-panel p-3 text-xs">
          <h4 className="mb-1 text-[11px] uppercase tracking-wider text-dim">Errors ({unit}{!metric ? `; body height ≈ ${p.body_px.toFixed(0)} px` : ""})</h4>
          {ev && !ev.included && (
            <p className="mb-2 rounded border border-warn/40 bg-warn/10 p-1 text-warn">Excluded from statistics at {h}s: {ev.exclusion_reasons?.join(", ")}</p>
          )}
          <table className="tbl">
            <thead><tr><th>model</th><th>ADE top-1 so far</th><th>FDE top-1 @ +{reveal.toFixed(1)}s</th><th>final ADE / FDE</th><th>best-of-K FDE*</th></tr></thead>
            <tbody>
              {summary.models.map((m) => {
                const mo = p.models[m];
                if (!mo) return null;
                const traj = m === "baseline" ? mo.trajectories.find((t) => t.id === "CV")! : [...mo.trajectories].sort((a, b) => a.rank - b.rank)[0];
                const part = partialErrors(traj.times, traj.points, actT, act, Math.min(reveal, h));
                const sc = ev?.included ? ev.scores?.[m] : undefined;
                const showFinal = reveal >= h - 1e-6 || endedAt !== null;
                return (
                  <tr key={m}>
                    <td>{MODEL_SHORT[m]}</td>
                    <td className="font-mono">{part ? fmt(part.ade) : "–"}</td>
                    <td className="font-mono">{part ? fmt(part.fde) : "–"}</td>
                    <td className="font-mono">{!showFinal ? "hidden until revealed" : sc ? `${fmt(sc.ade_top1)} / ${fmt(sc.fde_top1)}` : "excluded"}</td>
                    <td className="font-mono">{showFinal && sc ? `${fmt(sc.min_fde)} (K=${sc.k})` : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {(reveal >= h - 1e-6 || endedAt !== null) && ev?.included && (
            <div className="mt-2 space-y-0.5 text-[11px] text-dim">
              {summary.models.map((m) => {
                const sc = ev.scores?.[m];
                if (!sc) return null;
                return (
                  <p key={m}>
                    {MODEL_SHORT[m]}: top path {sc.top_id}{sc.top_prob !== null ? ` (${pctLabel(sc.top_prob, p.models[m]?.probability_type ?? "none")})` : ""}, closest path {sc.best_id} ·{" "}
                    {sc.miss_top1 ? <span className="text-bad">miss</span> : <span className="text-good">hit</span>} (threshold {fmt(sc.miss_threshold, metric ? 1 : 0)} {unit}) · actual end inside 50% / 90% region: {sc.in_hdr50 ? "yes" : "no"} / {sc.in_hdr90 ? "yes" : "no"}
                  </p>
                );
              })}
              <p>*Best-of-K picks the closest hypothesis with hindsight; the model did not select it in advance.</p>
            </div>
          )}
          <p className="mt-2 text-[11px] text-dim">
            Stored {p.causality.latest_frame_t_used.toFixed(2)} s · history {p.observation_window.steps} steps · tracker confidence {fmt(p.track_quality.tracker_confidence)}
            {p.camera_moving_at_t ? " · camera moving" : ""} · {Object.values(p.models).map((m) => `${m.model_version} (${m.probability_type})`).join(", ")}
          </p>
          {p.flags.length > 0 && <p className="text-[11px] text-warn">quality flags: {p.flags.join(", ")}</p>}
        </section>

        <section className="rounded-lg border border-line bg-panel p-3 text-xs">
          <h4 className="mb-1 text-[11px] uppercase tracking-wider text-dim">Review</h4>
          <input className="inp mb-2 w-full" placeholder="note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="flex gap-2">
            <button className="btn flex-1" onClick={() => flag("hard_case")}>Flag hard case</button>
            {trackFlagged ? (
              <button className="btn flex-1 text-warn" onClick={unflagTrack}>Undo: restore track #{p.track_id}</button>
            ) : (
              <button className="btn flex-1" onClick={() => flag("inaccurate_track")}>Flag track #{p.track_id} as inaccurate</button>
            )}
          </div>
          {trackFlagged && <p className="mt-1 text-[11px] text-warn">Track #{p.track_id} is flagged as inaccurate and excluded from every statistic.</p>}
          <p className="mt-1 text-[10px] text-dim">Inaccurate tracks are excluded when statistics are recomputed; hard cases go into the exported review set.</p>
          <div className="mt-2 border-t border-line pt-2">
            <label className="flex items-center gap-2"><input type="checkbox" checked={correcting} onChange={(e) => { setCorrecting(e.target.checked); setBrowseT(null); setPlaying(false); }} /> correct the actual future</label>
            {correcting && (
              <div className="mt-1 space-y-1">
                <p className="text-[10px] text-dim">
                  Move the reveal slider to a time, then click where person #{p.track_id} actually is on the video. Repeat for several times up to +{p.hmax} s.
                  {metric ? " Points are converted to ground-plane meters with the run's calibration." : " Points are stored in image pixels."}
                </p>
                <p className="font-mono text-[10px]">{corr.length ? corr.map((x) => `+${x.t.toFixed(2)}s`).join(" · ") : "no points yet"}</p>
                <div className="flex gap-2">
                  <button className="btn flex-1" disabled={!corr.length} onClick={() => setCorr((xs) => xs.slice(0, -1))}>Undo point</button>
                  <button className="btn flex-1" disabled={!corr.length} onClick={() => setCorr([])}>Clear</button>
                  <button className="btn btn-primary flex-1" disabled={!corr.length} onClick={saveCorrection}>Save correction</button>
                </div>
                <p className="text-[10px] text-dim">Corrections replace the tracker&apos;s path in the exported review set (future_source = user_correction). They do not change the statistics.</p>
              </div>
            )}
          </div>
          {msg && <p className="mt-1 text-[11px] text-dim">{msg}</p>}
        </section>
      </div>
    </div>
  );
}
