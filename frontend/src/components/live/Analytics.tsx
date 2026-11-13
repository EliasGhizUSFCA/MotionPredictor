"use client";
// Bird's-eye map, probability chart/table and kinematics time series for the live console.

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import TrajectoryMap, { type MapItem } from "@/components/ui/TrajectoryMap";
import type { FrameResult, Pt, TrackOut } from "@/lib/types";
import { OBS_COLOR, fmt, interpAt, rankColor } from "@/lib/viz";

const UI_H = [0.5, 1, 2, 3, 5];

function trackPaths(tr: TrackOut, metric: boolean) {
  const hist = metric ? tr.history_world : tr.history_px;
  return (hist ?? []).map((h) => [h[0], h[1]] as Pt);
}

export function BirdsEye({ result, selected, onSelect, highlight, setHighlight }: {
  result: FrameResult | null;
  selected: number | null;
  onSelect: (id: number) => void;
  highlight: string | null;
  setHighlight: (id: string | null) => void;
}) {
  const metric = result?.camera_calibration.coordinate_system === "ground_plane_meters";
  const items = useMemo(() => {
    const out: MapItem[] = [];
    if (!result) return out;
    for (const tr of result.tracks) {
      const sel = tr.track_id === selected;
      const hist = trackPaths(tr, metric);
      if (hist.length > 1) out.push({ kind: "line", pts: hist, color: OBS_COLOR, width: sel ? 3 : 2, id: String(tr.track_id) });
      const cur = hist.at(-1);
      if (cur) {
        out.push({ kind: "point", at: cur, color: sel ? "#38bdf8" : "#e2e8f0", radius: sel ? 5 : 4 });
        out.push({ kind: "label", at: cur, color: "#cbd5e1", text: `ID ${tr.track_id}` });
      }
      const f = tr.forecast;
      if (f && tr.state !== "lost") {
        for (const t of [...f.trajectories].sort((a, b) => b.rank - a.rank)) {
          const col = rankColor(t.rank);
          const dim = highlight && highlight !== t.id && sel ? 0.25 : 1;
          if (t.sigma.some((s) => s[0] > 0))
            t.points.forEach((p, i) => i % 3 === 0 && i > 0 && out.push({ kind: "ellipse", at: p, ellipse: t.sigma[i], color: col, width: 1, alpha: 0.35 * dim }));
          out.push({ kind: "line", pts: t.points, color: col, width: t.rank === 0 ? 2.5 : 1.8, dash: [6, 4], alpha: dim });
          const end = t.points.at(-1)!;
          if (sel) out.push({ kind: "label", at: end, color: col, text: t.probability === null ? t.id : `${t.id} ${(t.probability * 100).toFixed(0)}%${f.probability_type === "calibrated_probability" ? "" : " (uncal.)"}`, alpha: dim });
        }
      }
      const gt = result.replay_ground_truth?.[String(tr.track_id)];
      if (gt && sel && cur) out.push({ kind: "line", pts: [cur, ...gt.map((g) => [g[0], g[1]] as Pt)], color: "#ffffff", width: 1.5, dash: [2, 3], alpha: 0.8 });
    }
    return out;
  }, [result, selected, metric, highlight]);

  return (
    <div>
      <TrajectoryMap
        items={items}
        metric={metric}
        title={metric ? "Top-down ground plane (m) · observed = blue, predicted = dashed" : "Image-plane coordinates (px, NOT a top-down view - calibrate for metric bird's-eye)"}
        onPick={(id) => onSelect(Number(id))}
        fitKey={`${metric}-${result?.session_id}`}
      />
      {highlight && (
        <button className="mt-1 text-[11px] text-dim underline" onClick={() => setHighlight(null)}>
          clear trajectory highlight ({highlight})
        </button>
      )}
    </div>
  );
}

export function ProbabilityPanel({ track, setHighlight }: { track: TrackOut | null; setHighlight: (id: string) => void }) {
  const f = track?.forecast;
  if (!f) return <div className="text-sm text-dim">Select a confirmed track with a forecast.</div>;
  const metric = f.coordinate_system === "ground_plane_meters";
  const u = metric ? "m" : "px";
  const hasP = f.trajectories.some((t) => t.probability !== null);
  const data = f.trajectories.map((t) => ({ id: t.id, p: t.probability ?? 0, rank: t.rank }));
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2 text-[11px]">
        <span className="chip">{f.prediction_source === "trained_model" ? "learned model" : "physics baseline"}</span>
        <span className="chip">{f.model_version}</span>
        <span className={`chip ${f.probability_type === "calibrated_probability" ? "text-good" : "text-warn"}`} title={f.calibration_status}>
          {f.probability_type === "calibrated_probability" ? "calibrated probabilities (checkpoint held-out validated)" : f.probability_type === "none" ? "no route probabilities" : "uncalibrated weights"}
        </span>
        <span className="chip">{f.coordinate_system}</span>
        <span className="chip">history {f.history_steps_used} steps</span>
      </div>
      {f.calibration_status && f.probability_type !== "none" && <p className="text-[10px] text-dim">{f.calibration_status}</p>}
      {hasP ? (
        <div className="h-28">
          <ResponsiveContainer>
            <BarChart data={data} layout="vertical" margin={{ left: 4, right: 30 }}>
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#64748b" fontSize={10} />
              <YAxis type="category" dataKey="id" stroke="#64748b" fontSize={11} width={20} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ background: "#121924", border: "1px solid #273142" }} />
              <Bar dataKey="p" isAnimationActive animationDuration={250} onClick={(d: any) => setHighlight(d.id)}>
                {data.map((d) => (
                  <Cell key={d.id} fill={rankColor(d.rank)} cursor="pointer" />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="text-xs text-dim">The physics baseline extrapolates motion with an uncertainty band; it does not assign probabilities to routes.</p>
      )}
      {f.residual_probability !== null && f.residual_probability > 0.001 && (
        <p className="text-[11px] text-dim">Other hidden hypotheses: {(f.residual_probability * 100).toFixed(1)}%</p>
      )}
      <table className="tbl">
        <thead>
          <tr>
            <th>path</th>
            <th>{f.probability_type === "calibrated_probability" ? "prob." : "weight"}</th>
            {UI_H.filter((h) => h <= f.horizon_seconds + 1e-6).map((h) => (
              <th key={h}>
                @{h}s ({u}) ±1σ
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {f.trajectories.map((t) => (
            <tr key={t.id} onClick={() => setHighlight(t.id)} className="cursor-pointer hover:bg-panel2">
              <td style={{ color: rankColor(t.rank) }}>{t.id}</td>
              <td>{t.probability === null ? "–" : `${(t.probability * 100).toFixed(1)}%`}</td>
              {UI_H.filter((h) => h <= f.horizon_seconds + 1e-6).map((h) => {
                const p = interpAt(t.times, t.points, h);
                const i = t.times.findIndex((x) => x >= h - 1e-6);
                return (
                  <td key={h}>
                    {p ? `${fmt(p[0], metric ? 1 : 0)},${fmt(p[1], metric ? 1 : 0)}` : "–"}
                    {i >= 0 && t.sigma[i][0] > 0 ? ` ±${fmt(t.sigma[i][0], metric ? 2 : 0)}` : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-dim">Calibration status: {f.calibration_status}. {pathCountText(f.trajectories.length)} not a complete spatial probability density.</p>
      {f.warnings.map((w) => (
        <p key={w} className="text-[11px] text-warn">
          ⚠ {w}
        </p>
      ))}
    </div>
  );
}

const NUM_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"];
function pathCountText(n: number): string {
  const w = NUM_WORDS[n] ?? String(n);
  return n === 1 ? "A single discrete path is" : `${w} discrete paths are`;
}

export function KinematicsCharts({ history, trackId }: { history: FrameResult[]; trackId: number | null }) {
  const [which, setWhich] = useState<"speed" | "accel" | "heading" | "turn">("speed");
  // `history` is mutated in place by useStream (push/splice), so its identity never changes while
  // streaming. Key the memo on the latest frame as well, or the chart freezes after the first render.
  const latest = history.at(-1);
  const version = `${history.length}:${latest?.session_id ?? ""}:${latest?.frame_id ?? -1}:${latest?.server_done_ms ?? 0}`;
  const series = useMemo(() => {
    if (trackId === null || !history.length) return [];
    const t0 = history.at(-1)!.server_done_ms;
    const out: { t: number; v: number }[] = [];
    let prevH: number | null = null;
    let unwrap = 0;
    for (const r of history) {
      const tr = r.tracks.find((x) => x.track_id === trackId);
      const k = tr?.kinematics_world ?? tr?.kinematics;
      if (!k) continue;
      let v = 0;
      if (which === "speed") v = k.speed;
      else if (which === "accel") v = k.accel_magnitude;
      else if (which === "turn") v = k.turn_rate_deg_s;
      else {
        if (prevH !== null) {
          const d = k.heading_deg - prevH;
          if (d > 180) unwrap -= 360;
          else if (d < -180) unwrap += 360;
        }
        prevH = k.heading_deg;
        v = k.heading_deg + unwrap;
      }
      out.push({ t: (r.server_done_ms - t0) / 1000, v });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, version, trackId, which]);
  const last = history.at(-1)?.tracks.find((x) => x.track_id === trackId);
  const units = (last?.kinematics_world ?? last?.kinematics)?.units === "m" ? "m" : "px";
  const unit = { speed: `${units}/s`, accel: `${units}/s²`, heading: "deg", turn: "deg/s" }[which];
  return (
    <div>
      <div className="mb-1 flex gap-1">
        {(["speed", "accel", "heading", "turn"] as const).map((w) => (
          <button key={w} onClick={() => setWhich(w)} className={`tab ${which === w ? "tab-on" : ""}`}>
            {w}
          </button>
        ))}
        <span className="ml-auto text-[11px] text-dim">{trackId === null ? "select a track" : `ID ${trackId} · ${unit} · Kalman-filtered`}</span>
      </div>
      <div className="h-36">
        <ResponsiveContainer>
          <LineChart data={series} margin={{ left: -10, right: 8, top: 4 }}>
            <CartesianGrid stroke="#16202c" />
            <XAxis dataKey="t" type="number" domain={["dataMin", 0]} tickFormatter={(v) => `${v.toFixed(0)}s`} stroke="#64748b" fontSize={10} />
            <YAxis stroke="#64748b" fontSize={10} />
            <Tooltip formatter={(v: number) => v.toFixed(2)} labelFormatter={(v) => `${Number(v).toFixed(1)} s`} contentStyle={{ background: "#121924", border: "1px solid #273142" }} />
            <Line dataKey="v" stroke={OBS_COLOR} dot={false} isAnimationActive={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
