"use client";
// Pipeline observability: one card per stage, built only from events the backend measured
// (plus the browser's own overlay timing). No narrative, no hidden "reasoning".

import { useRef, useState } from "react";
import type { FrameResult, StageEvent } from "@/lib/types";

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const max = Math.max(...values, 1e-3);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 60},${16 - (v / max) * 15}`).join(" ");
  return (
    <svg width="60" height="16" className="shrink-0">
      <polyline points={pts} fill="none" stroke="#38bdf8" strokeWidth="1.2" />
    </svg>
  );
}

function Card({ ev, hist }: { ev: StageEvent; hist: number[] }) {
  const [open, setOpen] = useState(false);
  const color = ev.status === "error" ? "bg-bad" : ev.status === "skipped" ? "bg-dim" : "bg-good";
  return (
    <div className="rounded border border-line bg-panel2">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-2 py-1.5 text-left">
        <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />
        <span className="flex-1 truncate text-xs text-ink">{ev.label}</span>
        <Spark values={hist} />
        <span className="w-16 text-right font-mono text-[11px] text-dim">{ev.status === "skipped" ? "skipped" : `${ev.duration_ms.toFixed(1)} ms`}</span>
      </button>
      {open && (
        <div className="space-y-1 border-t border-line px-2 py-1.5 text-[11px]">
          {ev.explanation && <p className="text-dim">{ev.explanation}</p>}
          <p className="font-mono text-dim">t = {new Date(ev.t_start_ms).toISOString().slice(11, 23)}{ev.confidence !== null ? ` · mean confidence ${ev.confidence.toFixed(3)}` : ""}</p>
          {Object.keys(ev.inputs).length > 0 && (
            <>
              <p className="text-ink">inputs</p>
              <pre className="max-h-32 overflow-auto rounded bg-bg p-1 text-[10px] text-dim">{JSON.stringify(ev.inputs, null, 1)}</pre>
            </>
          )}
          <p className="text-ink">outputs</p>
          <pre className="max-h-48 overflow-auto rounded bg-bg p-1 text-[10px] text-dim">{JSON.stringify(ev.outputs, null, 1)}</pre>
          {ev.error && <p className="text-bad">error: {ev.error}</p>}
        </div>
      )}
    </div>
  );
}

export default function Diagnostics({ result, renderMs }: { result: FrameResult | null; renderMs: number | null }) {
  const hist = useRef(new Map<string, number[]>());
  const lastFrame = useRef<number>(-1);
  if (!result) return <div className="text-sm text-dim">Start a source to see live pipeline events.</div>;
  const events: StageEvent[] = [
    ...result.events,
    {
      stage: "overlay", label: "Overlay rendered (browser)", t_start_ms: Date.now(), duration_ms: renderMs ?? 0, status: "ok",
      inputs: { tracks: result.tracks.length }, outputs: { canvas_draw_ms: renderMs }, confidence: null,
      explanation: "Measured in the browser: time to draw boxes, trails and forecast paths for the latest result.", error: null,
    },
  ];
  if (lastFrame.current !== result.frame_id) {
    lastFrame.current = result.frame_id;
    for (const e of events) {
      const h = hist.current.get(e.stage) ?? [];
      h.push(e.duration_ms);
      if (h.length > 60) h.shift();
      hist.current.set(e.stage, h);
    }
  }
  const total = result.events.reduce((s, e) => s + e.duration_ms, 0);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[11px] text-dim">
        <span>frame #{result.frame_id} · {result.source}</span>
        <span>
          stages {total.toFixed(1)} ms · end-to-end server {result.processing_ms.toFixed(1)} ms
          {result.stats.compute_ms_mean != null ? ` · compute ${result.stats.compute_ms_mean.toFixed(1)} ms` : ""}
          {result.stats.queue_ms_mean != null ? ` · queue ${result.stats.queue_ms_mean.toFixed(1)} ms (means)` : ""}
        </span>
      </div>
      {events.map((e) => (
        <Card key={e.stage} ev={e} hist={hist.current.get(e.stage) ?? []} />
      ))}
      {result.warnings.length > 0 && (
        <div className="space-y-1 pt-1">
          {result.warnings.map((w) => (
            <p key={w} className="rounded border border-warn/40 bg-warn/10 px-2 py-1 text-[11px] text-warn">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
