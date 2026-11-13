// Shared drawing helpers and colour conventions.
// Predictions are dashed; observations are solid blue. Rank colours: green, yellow, red/pink.

import type { MaybePt, Pt } from "./types";

export const RANK_COLORS = ["#22c55e", "#eab308", "#f43f5e", "#e879f9", "#fb923c"];
export const OBS_COLOR = "#3b82f6";
export const ACTUAL_COLOR = "#3b82f6";

export function rankColor(rank: number): string {
  return RANK_COLORS[Math.min(rank, RANK_COLORS.length - 1)];
}

export function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function valid(p: MaybePt | Pt | undefined | null): p is Pt {
  return !!p && p[0] !== null && p[1] !== null && Number.isFinite(p[0] as number) && Number.isFinite(p[1] as number);
}

/** Draw a polyline, skipping invalid points (e.g. ground points projected beyond the horizon). */
export function polyline(ctx: CanvasRenderingContext2D, pts: (MaybePt | Pt)[], map: (p: Pt) => Pt = (p) => p) {
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    if (!valid(p)) {
      pen = false;
      continue;
    }
    const [x, y] = map(p);
    if (pen) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
    pen = true;
  }
  ctx.stroke();
}

/** Closed band polygon left[] + reversed right[]. */
export function band(ctx: CanvasRenderingContext2D, left: MaybePt[], right: MaybePt[], map: (p: Pt) => Pt = (p) => p) {
  const l = left.filter(valid).map(map);
  const r = right.filter(valid).map(map).reverse();
  if (l.length < 2 || r.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(l[0][0], l[0][1]);
  for (const p of [...l.slice(1), ...r]) ctx.lineTo(p[0], p[1]);
  ctx.closePath();
  ctx.fill();
}

export function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, dx: number, dy: number, head = 7) {
  const len = Math.hypot(dx, dy);
  if (len < 1) return;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + dx, y + dy);
  ctx.stroke();
  const a = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x + dx, y + dy);
  ctx.lineTo(x + dx - head * Math.cos(a - 0.4), y + dy - head * Math.sin(a - 0.4));
  ctx.lineTo(x + dx - head * Math.cos(a + 0.4), y + dy - head * Math.sin(a + 0.4));
  ctx.closePath();
  ctx.fill();
}

// COCO-17 skeleton edges
export const SKELETON: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

export function fmt(v: number | null | undefined, d = 2): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(d);
}

export function probLabel(p: number | null, type: string): string {
  if (p === null) return "no probability";
  const s = `${(p * 100).toFixed(0)}%`;
  return type === "calibrated_probability" ? s : `${s} (uncal. weight)`;
}

/** Linear interpolation of a polyline sampled at `times` at time `t`. */
export function interpAt(times: number[], pts: Pt[], t: number): Pt | null {
  if (!pts.length) return null;
  if (t <= times[0]) return pts[0];
  for (let i = 1; i < times.length; i++) {
    if (t <= times[i]) {
      const a = (t - times[i - 1]) / Math.max(times[i] - times[i - 1], 1e-9);
      return [pts[i - 1][0] + a * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + a * (pts[i][1] - pts[i - 1][1])];
    }
  }
  return null; // beyond available data
}

/** ADE/FDE of a predicted path against an observed path up to time `upTo` (inclusive), sampled at prediction times. */
export function partialErrors(predT: number[], pred: Pt[], actT: number[], act: Pt[], upTo: number): { ade: number; fde: number; n: number } | null {
  const errs: number[] = [];
  for (let i = 0; i < predT.length; i++) {
    const t = predT[i];
    if (t <= 1e-6 || t > upTo + 1e-6) continue;
    const a = interpAt(actT, act, t);
    if (!a) continue;
    errs.push(Math.hypot(pred[i][0] - a[0], pred[i][1] - a[1]));
  }
  if (!errs.length) return null;
  return { ade: errs.reduce((s, x) => s + x, 0) / errs.length, fde: errs[errs.length - 1], n: errs.length };
}

/** Cross-track ±k·sigma boundary of a path from per-point ellipse params (port of pipeline.band_points). */
export function bandFromSigma(points: Pt[], sigma: [number, number, number][], k = 2): { left: Pt[]; right: Pt[] } {
  const left: Pt[] = [];
  const right: Pt[] = [];
  for (let i = 0; i < points.length; i++) {
    const a = points[Math.max(i - 1, 0)];
    const b = points[Math.min(i + 1, points.length - 1)];
    let nx = -(b[1] - a[1]);
    let ny = b[0] - a[0];
    const nn = Math.hypot(nx, ny);
    if (nn < 1e-9) [nx, ny] = [0, 1];
    else [nx, ny] = [nx / nn, ny / nn];
    const [sa, sb, ang] = sigma[i] ?? [0, 0, 0];
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    // n^T R diag(a²,b²) R^T n
    const u = nx * c + ny * s;
    const v = -nx * s + ny * c;
    const sd = Math.sqrt(u * u * sa * sa + v * v * sb * sb);
    left.push([points[i][0] + k * sd * nx, points[i][1] + k * sd * ny]);
    right.push([points[i][0] - k * sd * nx, points[i][1] - k * sd * ny]);
  }
  return { left, right };
}

/** Apply a 3x3 homography (row-major) to a point; returns null when the point is behind the horizon. */
export function applyH(Hm: number[][], p: Pt, sign = 1): Pt | null {
  const x = Hm[0][0] * p[0] + Hm[0][1] * p[1] + Hm[0][2];
  const y = Hm[1][0] * p[0] + Hm[1][1] * p[1] + Hm[1][2];
  const w = Hm[2][0] * p[0] + Hm[2][1] * p[1] + Hm[2][2];
  if (Math.abs(w) < 1e-12 || Math.sign(w) !== sign) return null;
  return [x / w, y / w];
}

/** Horizon key as written by the Python backend (str(float)): 3 -> "3.0", 0.5 -> "0.5". */
export function hk(h: number): string {
  return Number.isInteger(h) ? h.toFixed(1) : String(h);
}
