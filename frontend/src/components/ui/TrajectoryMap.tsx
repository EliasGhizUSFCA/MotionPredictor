"use client";
// Generic 2-D trajectory map with zoom (wheel), pan (drag), fit, grid and scale bar.
// Works in metric ground-plane coordinates (y up) or image-plane pixels (y down, clearly labelled).

import { useCallback, useEffect, useRef, useState } from "react";
import type { Pt } from "@/lib/types";
import { rgba } from "@/lib/viz";

export interface MapItem {
  kind: "line" | "band" | "point" | "ellipse" | "label";
  pts?: Pt[];
  left?: Pt[];
  right?: Pt[];
  at?: Pt;
  color: string;
  width?: number;
  dash?: number[];
  alpha?: number;
  radius?: number;
  text?: string;
  ellipse?: [number, number, number]; // sigma major, minor, angle
  id?: string; // clickable identity
}

interface Props {
  items: MapItem[];
  metric: boolean;
  title: string;
  height?: number;
  onPick?: (id: string) => void;
  fitKey?: string; // refit when this changes
}

export default function TrajectoryMap({ items, metric, title, height = 280, onPick, fitKey }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState({ cx: 0, cy: 0, scale: 20 }); // px per unit
  const drag = useRef<{ x: number; y: number; cx: number; cy: number } | null>(null);
  const fitted = useRef<string | undefined>(undefined);

  const fit = useCallback(() => {
    const pts: Pt[] = [];
    for (const it of items) {
      if (it.pts) pts.push(...it.pts);
      if (it.at) pts.push(it.at);
    }
    const c = ref.current;
    if (!pts.length || !c) return;
    const xs = pts.map((p) => p[0]).filter(Number.isFinite);
    const ys = pts.map((p) => p[1]).filter(Number.isFinite);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const w = c.clientWidth;
    const h = c.clientHeight;
    const span = Math.max((x1 - x0) / w, (y1 - y0) / h, metric ? 0.02 : 0.2);
    setView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: 0.8 / span });
  }, [items, metric]);

  useEffect(() => {
    if (fitKey !== fitted.current && items.length) {
      fitted.current = fitKey;
      fit();
    }
  }, [fitKey, items, fit]);

  const toScreen = useCallback(
    (p: Pt, w: number, h: number): Pt => {
      const x = (p[0] - view.cx) * view.scale + w / 2;
      const y = metric ? -(p[1] - view.cy) * view.scale + h / 2 : (p[1] - view.cy) * view.scale + h / 2;
      return [x, y];
    },
    [view, metric],
  );

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    c.width = w * dpr;
    c.height = h * dpr;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, w, h);
    // grid at a "nice" spacing
    const target = 60 / view.scale;
    const pow = 10 ** Math.floor(Math.log10(target));
    const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= target) ?? pow * 10;
    ctx.strokeStyle = "#16202c";
    ctx.lineWidth = 1;
    const xMin = view.cx - w / 2 / view.scale;
    const xMax = view.cx + w / 2 / view.scale;
    const yMin = view.cy - h / 2 / view.scale;
    const yMax = view.cy + h / 2 / view.scale;
    for (let gx = Math.ceil(xMin / step) * step; gx <= xMax; gx += step) {
      const [sx] = toScreen([gx, 0], w, h);
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, h);
      ctx.stroke();
    }
    for (let gy = Math.ceil(yMin / step) * step; gy <= yMax; gy += step) {
      const [, sy] = toScreen([0, gy], w, h);
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(w, sy);
      ctx.stroke();
    }
    const S = (p: Pt) => toScreen(p, w, h);
    for (const it of items) {
      ctx.globalAlpha = it.alpha ?? 1;
      ctx.strokeStyle = it.color;
      ctx.fillStyle = it.color;
      ctx.lineWidth = it.width ?? 2;
      ctx.setLineDash(it.dash ?? []);
      if (it.kind === "line" && it.pts && it.pts.length > 1) {
        ctx.beginPath();
        it.pts.forEach((p, i) => {
          const [x, y] = S(p);
          if (i) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        ctx.stroke();
      } else if (it.kind === "band" && it.left && it.right && it.left.length > 1) {
        ctx.fillStyle = rgba(it.color, it.alpha ?? 0.15);
        ctx.globalAlpha = 1;
        ctx.beginPath();
        [...it.left, ...[...it.right].reverse()].forEach((p, i) => {
          const [x, y] = S(p);
          if (i) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
      } else if (it.kind === "point" && it.at) {
        const [x, y] = S(it.at);
        ctx.beginPath();
        ctx.arc(x, y, it.radius ?? 4, 0, Math.PI * 2);
        ctx.fill();
      } else if (it.kind === "ellipse" && it.at && it.ellipse) {
        const [x, y] = S(it.at);
        const [a, b, ang] = it.ellipse;
        ctx.beginPath();
        ctx.ellipse(x, y, Math.max(a * view.scale, 0.5), Math.max(b * view.scale, 0.5), metric ? -ang : ang, 0, Math.PI * 2);
        ctx.stroke();
      } else if (it.kind === "label" && it.at && it.text) {
        const [x, y] = S(it.at);
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillText(it.text, x + 6, y - 6);
      }
    }
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    // scale bar
    ctx.fillStyle = "#94a3b8";
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(10, h - 12);
    ctx.lineTo(10 + step * view.scale, h - 12);
    ctx.stroke();
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`${step} ${metric ? "m" : "px"}`, 12, h - 16);
  }, [items, view, toScreen, metric]);

  const onWheel = (e: React.WheelEvent) => {
    const k = Math.exp(-e.deltaY * 0.0015);
    setView((v) => ({ ...v, scale: Math.min(Math.max(v.scale * k, 0.01), 5000) }));
  };
  const onDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy };
  };
  const onMove = (e: React.MouseEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = (e.clientX - d.x) / view.scale;
    const dy = (e.clientY - d.y) / view.scale;
    setView((v) => ({ ...v, cx: d.cx - dx, cy: metric ? d.cy + dy : d.cy - dy }));
  };
  const onUp = (e: React.MouseEvent) => {
    const d = drag.current;
    drag.current = null;
    if (!d || !onPick || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 3) return;
    const c = ref.current!;
    const r = c.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    let best: { id: string; d: number } | null = null;
    for (const it of items) {
      if (!it.id || !it.pts) continue;
      for (const p of it.pts) {
        const [x, y] = toScreen(p, c.clientWidth, c.clientHeight);
        const dd = Math.hypot(x - mx, y - my);
        if (dd < 12 && (!best || dd < best.d)) best = { id: it.id, d: dd };
      }
    }
    if (best) onPick(best.id);
  };

  return (
    <div className="relative">
      <div className="absolute left-2 top-1 z-10 text-[11px] text-dim">{title}</div>
      <button onClick={fit} className="absolute right-2 top-1 z-10 rounded border border-line bg-panel px-2 text-[11px] text-dim hover:text-ink">
        fit
      </button>
      <canvas
        ref={ref}
        className="w-full cursor-grab rounded border border-line active:cursor-grabbing"
        style={{ height }}
        onWheel={onWheel}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => (drag.current = null)}
      />
    </div>
  );
}
