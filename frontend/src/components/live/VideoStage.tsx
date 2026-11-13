"use client";
import { useEffect, useRef } from "react";
import type { FrameResult, Pt, TrackOut } from "@/lib/types";
import { OBS_COLOR, SKELETON, arrow, band, polyline, probLabel, rankColor, rgba, valid } from "@/lib/viz";

export interface Layers {
  pose: boolean;
  paths: boolean;
  cones: boolean;
  trails: boolean;
  truth: boolean;
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  result: FrameResult | null;
  demoFrame: string | null;
  layers: Layers;
  selected: number | null;
  onSelect: (id: number | null) => void;
  showVideo: boolean;
  onRenderMs?: (ms: number) => void;
}

/** Ground (meters) -> frame pixels with the inverse homography of the normalised-image calibration. */
export function groundToPx(Hinv: number[][], p: Pt, W: number, H: number): Pt | null {
  const x = Hinv[0][0] * p[0] + Hinv[0][1] * p[1] + Hinv[0][2];
  const y = Hinv[1][0] * p[0] + Hinv[1][1] * p[1] + Hinv[1][2];
  const w = Hinv[2][0] * p[0] + Hinv[2][1] * p[1] + Hinv[2][2];
  if (Math.abs(w) < 1e-12) return null;
  return [(x / w) * W, (y / w) * H];
}

export default function VideoStage({ videoRef, result, demoFrame, layers, selected, onSelect, showVideo, onRenderMs }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const smooth = useRef(new Map<number, number[]>());
  const latest = useRef({ result, layers, selected });
  latest.current = { result, layers, selected };

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const c = canvas.current;
      const { result: r, layers: L, selected: sel } = latest.current;
      if (!c) return;
      const rect = c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(rect.width * dpr)) {
        c.width = Math.round(rect.width * dpr);
        c.height = Math.round(rect.height * dpr);
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, c.width, c.height);
      if (!r) return;
      const t0 = performance.now();
      const sx = c.width / r.frame_width;
      const sy = c.height / r.frame_height;
      ctx.setTransform(sx, 0, 0, sy, 0, 0);
      const u = 1 / sx; // one screen pixel in frame units
      const age = performance.now() - ((r as any)._receivedAt ?? performance.now());
      const stale = age > 1000;

      for (const tr of r.tracks) drawTrack(ctx, tr, r, L, tr.track_id === sel, u, stale);

      // prediction age (never hide stale predictions - label them)
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.font = `${12 * dpr}px ui-monospace, monospace`;
      ctx.fillStyle = stale ? "#f43f5e" : "rgba(214,221,230,0.85)";
      const ageTxt = `result age ${age.toFixed(0)} ms · frame #${r.frame_id}${stale ? " · STALE" : ""}`;
      ctx.fillText(ageTxt, 8 * dpr, c.height - 8 * dpr);
      onRenderMs?.(performance.now() - t0);
    };

    const drawTrack = (ctx: CanvasRenderingContext2D, tr: TrackOut, r: FrameResult, L: Layers, isSel: boolean, u: number, stale: boolean) => {
      // smooth box display (exponential), keyed by track id
      const prev = smooth.current.get(tr.track_id);
      const b = prev ? prev.map((v, i) => v + 0.5 * (tr.bbox[i] - v)) : [...tr.bbox];
      smooth.current.set(tr.track_id, b);
      const lost = tr.state === "lost";
      const base = lost ? "#64748b" : isSel ? "#38bdf8" : "#22d3ee";
      ctx.lineWidth = (isSel ? 2.5 : 1.5) * u;
      ctx.strokeStyle = base;
      ctx.setLineDash(lost ? [6 * u, 4 * u] : []);
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.setLineDash([]);

      // trail (observed history)
      if (L.trails && tr.history_px.length > 1) {
        ctx.strokeStyle = rgba(OBS_COLOR, 0.9);
        ctx.lineWidth = 2.5 * u;
        polyline(ctx, tr.history_px.map((h) => [h[0], h[1]] as Pt));
      }

      // pose
      if (L.pose && tr.keypoints) {
        ctx.strokeStyle = "rgba(167,243,208,0.85)";
        ctx.lineWidth = 1.5 * u;
        for (const [a, c2] of SKELETON) {
          const p = tr.keypoints[a];
          const q = tr.keypoints[c2];
          if (p[2] > 0.4 && q[2] > 0.4) {
            ctx.beginPath();
            ctx.moveTo(p[0], p[1]);
            ctx.lineTo(q[0], q[1]);
            ctx.stroke();
          }
        }
        ctx.fillStyle = "#a7f3d0";
        for (const k of tr.keypoints) if (k[2] > 0.4) ctx.fillRect(k[0] - 1.5 * u, k[1] - 1.5 * u, 3 * u, 3 * u);
      }

      // forecasts
      const f = tr.forecast;
      if (f && L.paths && !lost) {
        const trajs = [...f.trajectories].sort((a, c2) => c2.rank - a.rank);
        for (const t of trajs) {
          const col = rankColor(t.rank);
          if (L.cones && t.band_left && t.band_right && t.sigma.some((s) => s[0] > 0)) {
            ctx.fillStyle = rgba(col, stale ? 0.05 : 0.13);
            band(ctx, t.band_left, t.band_right);
          }
          ctx.strokeStyle = rgba(col, stale ? 0.35 : 0.95);
          ctx.lineWidth = (t.rank === 0 ? 3 : 2) * u;
          ctx.setLineDash([8 * u, 5 * u]);
          if (t.image_points) polyline(ctx, t.image_points);
          ctx.setLineDash([]);
          const end = t.image_points?.filter(valid).at(-1);
          if (end) {
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(end[0] as number, end[1] as number, 3.5 * u, 0, Math.PI * 2);
            ctx.fill();
            ctx.font = `${11 * u}px ui-monospace, monospace`;
            ctx.fillText(`${t.id} ${t.probability === null ? "" : probLabel(t.probability, f.probability_type)}`, (end[0] as number) + 5 * u, (end[1] as number) - 4 * u);
          }
        }
      }

      // true future (dataset replay only, revealed for comparison)
      const gt = r.replay_ground_truth?.[String(tr.track_id)];
      const Hinv = r.camera_calibration.homography_inv;
      if (L.truth && gt && gt.length > 1 && Hinv) {
        const pts = gt.map((g) => groundToPx(Hinv, [g[0], g[1]], r.frame_width, r.frame_height)).filter(Boolean) as Pt[];
        ctx.strokeStyle = "rgba(255,255,255,0.75)";
        ctx.lineWidth = 1.5 * u;
        ctx.setLineDash([2 * u, 3 * u]);
        polyline(ctx, [tr.footpoint_px, ...pts]);
        ctx.setLineDash([]);
      }

      // footpoint + velocity + heading
      const [fx, fy] = tr.footpoint_px;
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(fx, fy, 3.5 * u, 0, Math.PI * 2);
      ctx.fill();
      const k = tr.kinematics;
      if (k && !lost) {
        ctx.strokeStyle = OBS_COLOR;
        ctx.fillStyle = OBS_COLOR;
        ctx.lineWidth = 2.5 * u;
        arrow(ctx, fx, fy, k.velocity[0] * 0.5, k.velocity[1] * 0.5, 8 * u); // 0.5 s of motion
        if (k.movement_state !== "stationary") {
          const h = (k.heading_deg * Math.PI) / 180;
          ctx.strokeStyle = "#e2e8f0";
          ctx.fillStyle = "#e2e8f0";
          ctx.lineWidth = 1.5 * u;
          arrow(ctx, fx, fy + 10 * u, Math.cos(h) * 22 * u, Math.sin(h) * 22 * u, 5 * u);
        }
      }

      // label
      const kw = tr.kinematics_world ?? tr.kinematics;
      const spd = kw ? `${kw.speed.toFixed(kw.units === "m" ? 2 : 0)} ${kw.units === "m" ? "m/s" : "px/s"}` : "";
      const ht = tr.height.status === "calibrated" && tr.height.meters ? `${tr.height.meters.toFixed(2)} m` : "height ?";
      const label = `ID ${tr.track_id} ${tr.state === "confirmed" ? "" : tr.state} · ${(tr.det_confidence * 100).toFixed(0)}% · ${spd} · ${ht}`;
      ctx.font = `${12 * u}px ui-monospace, monospace`;
      const w = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(11,15,20,0.8)";
      ctx.fillRect(b[0], b[1] - 16 * u, w + 8 * u, 16 * u);
      ctx.fillStyle = base;
      ctx.fillText(label, b[0] + 4 * u, b[1] - 4 * u);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [onRenderMs]);

  const aspect = result ? result.frame_width / result.frame_height : 16 / 9;

  const click = (e: React.MouseEvent) => {
    if (!result) return;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * result.frame_width;
    const y = ((e.clientY - rect.top) / rect.height) * result.frame_height;
    const hit = result.tracks.find((t) => x >= t.bbox[0] && x <= t.bbox[2] && y >= t.bbox[1] && y <= t.bbox[3]);
    onSelect(hit ? hit.track_id : null);
  };

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-line bg-black" style={{ aspectRatio: String(aspect) }}>
      <video ref={videoRef} className={`absolute inset-0 h-full w-full object-fill ${showVideo ? "" : "invisible"}`} muted playsInline />
      {demoFrame && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={demoFrame} alt="dataset replay frame (synthetic top-down rendering)" className="absolute inset-0 h-full w-full object-fill" />
      )}
      <canvas ref={canvas} className="absolute inset-0 h-full w-full cursor-crosshair" onClick={click} />
    </div>
  );
}
