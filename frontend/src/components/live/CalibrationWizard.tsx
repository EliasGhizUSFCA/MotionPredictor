"use client";
// Step-by-step ground-plane calibration. Points are stored in normalised image coordinates,
// so the calibration stays valid when the streaming resolution changes.

import { useEffect, useRef, useState } from "react";
import { postJSON } from "@/lib/api";
import type { CalibrationRequest } from "@/lib/types";

type Step = 1 | 2 | 3 | 4;

export default function CalibrationWizard({ snapshot, onApply, onClose, current }: {
  snapshot: string | null;
  onApply: (c: CalibrationRequest | null) => void;
  onClose: () => void;
  current: CalibrationRequest | null;
}) {
  const [step, setStep] = useState<Step>(1);
  const [pts, setPts] = useState<number[][]>(current?.image_points ?? []);
  const [world, setWorld] = useState<string[][]>(current?.world_points.map((w) => w.map(String)) ?? []);
  const [rect, setRect] = useState({ w: "3", d: "4" });
  const [heightMode, setHeightMode] = useState<"none" | "camera" | "reference">(current?.camera_height_m ? "camera" : current?.reference ? "reference" : "none");
  const [camH, setCamH] = useState(String(current?.camera_height_m ?? "2.5"));
  const [ref, setRef] = useState<{ foot?: number[]; top?: number[]; h: string }>({ foot: current?.reference?.foot, top: current?.reference?.top, h: String(current?.reference?.height_m ?? "2.0") });
  const [fit, setFit] = useState<Record<string, any> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);

  const req = (): CalibrationRequest => ({
    image_points: pts,
    world_points: world.map((w) => w.map(Number)),
    camera_height_m: heightMode === "camera" ? Number(camH) : null,
    reference: heightMode === "reference" && ref.foot && ref.top ? { foot: ref.foot, top: ref.top, height_m: Number(ref.h) } : null,
  });

  const onClick = (e: React.MouseEvent) => {
    const r = (e.target as HTMLElement).getBoundingClientRect();
    const p = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
    if (step === 2) {
      setPts((x) => [...x, p]);
      setWorld((w) => [...w, ["", ""]]);
    } else if (step === 3 && heightMode === "reference") {
      setRef((x) => (!x.foot || x.top ? { ...x, foot: p, top: undefined } : { ...x, top: p }));
    }
  };

  const fillRect = () => {
    const W = rect.w;
    const D = rect.d;
    setWorld((w) => w.map((row, i) => (i < 4 ? [[ "0", "0" ], [W, "0"], [W, D], ["0", D]][i] : row)));
  };

  const doFit = async () => {
    setErr(null);
    try {
      setFit(await postJSON("/api/calibration/fit", req()));
      setStep(4);
    } catch (e: any) {
      setErr(String(e.message ?? e));
    }
  };

  // draw points, horizon and a projected 1 m ground grid for visual validation
  useEffect(() => {
    const c = cv.current;
    const img = imgRef.current;
    if (!c || !img) return;
    const w = img.clientWidth;
    const h = img.clientHeight;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    if (fit?.homography_inv && step === 4) {
      const Hi = fit.homography_inv as number[][];
      const wp = (fit.world_points as number[][]) ?? [];
      // homographies are defined up to scale: visible ground has the same sign as a marked point
      const refSign = Math.sign(Hi[2][0] * wp[0][0] + Hi[2][1] * wp[0][1] + Hi[2][2]);
      const proj = (x: number, y: number) => {
        const u = Hi[0][0] * x + Hi[0][1] * y + Hi[0][2];
        const v = Hi[1][0] * x + Hi[1][1] * y + Hi[1][2];
        const s = Hi[2][0] * x + Hi[2][1] * y + Hi[2][2];
        return Math.sign(s) === refSign && Math.abs(s) > 1e-9 ? [(u / s) * w, (v / s) * h] : null;
      };
      const xs = wp.map((p) => p[0]);
      const ys = wp.map((p) => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs) - 3, Math.max(...xs) + 3, Math.min(...ys) - 3, Math.max(...ys) + 6];
      ctx.strokeStyle = "rgba(34,211,238,0.5)";
      for (let x = Math.floor(x0); x <= x1; x++) {
        ctx.beginPath();
        for (let y = y0; y <= y1; y += 0.25) {
          const p = proj(x, y);
          if (p) ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
      }
      for (let y = Math.floor(y0); y <= y1; y++) {
        ctx.beginPath();
        for (let x = x0; x <= x1; x += 0.25) {
          const p = proj(x, y);
          if (p) ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
      }
      const hz = fit.horizon_v as [number | null, number | null];
      if (hz[0] !== null && hz[1] !== null) {
        ctx.strokeStyle = "#f59e0b";
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(0, hz[0] * h);
        ctx.lineTo(w, hz[1] * h);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#f59e0b";
        ctx.fillText("horizon", 6, hz[0] * h - 4);
      }
    }
    pts.forEach((p, i) => {
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.arc(p[0] * w, p[1] * h, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.fillText(String(i + 1), p[0] * w + 7, p[1] * h - 7);
    });
    if (ref.foot) {
      ctx.strokeStyle = "#e879f9";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(ref.foot[0] * w, ref.foot[1] * h, 4, 0, Math.PI * 2);
      ctx.stroke();
      if (ref.top) {
        ctx.beginPath();
        ctx.moveTo(ref.foot[0] * w, ref.foot[1] * h);
        ctx.lineTo(ref.top[0] * w, ref.top[1] * h);
        ctx.stroke();
      }
    }
  }, [pts, fit, step, ref, snapshot]);

  const worldOk = world.length >= 4 && world.every((r) => r.every((v) => v !== "" && Number.isFinite(Number(v))));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="max-h-full w-full max-w-5xl overflow-auto rounded-lg border border-line bg-panel p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg text-ink">Camera calibration wizard · step {step}/4</h2>
          <button onClick={onClose} className="btn">close</button>
        </div>
        <div className="grid gap-4 md:grid-cols-[1fr_300px]">
          <div className="relative">
            {snapshot ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img ref={imgRef} src={snapshot} alt="calibration snapshot" className="w-full rounded border border-line" onLoad={() => setPts((p) => [...p])} />
            ) : (
              <div className="rounded border border-line p-8 text-dim">Start a camera or video first; the wizard uses a still frame.</div>
            )}
            <canvas ref={cv} className="absolute inset-0 h-full w-full cursor-crosshair" onClick={onClick} />
          </div>
          <div className="space-y-3 text-sm">
            {step === 1 && (
              <>
                <p className="text-dim">
                  Calibration maps the image onto the floor so the system can report positions and speeds in meters, and (optionally) approximate heights.
                  You need at least 4 points on the floor whose real positions you know - e.g. the corners of a rug, floor tiles or a taped rectangle you measured.
                  Use 5+ points to let the system measure its own error.
                </p>
                <p className="text-dim">Without calibration everything still works in image pixels, clearly labelled.</p>
                <button className="btn btn-primary" onClick={() => setStep(2)} disabled={!snapshot}>Begin</button>
                {current && <button className="btn" onClick={() => { onApply(null); onClose(); }}>Remove current calibration</button>}
              </>
            )}
            {step === 2 && (
              <>
                <p className="text-dim">Click floor points on the image, then enter their positions in meters (x to the right, y away from the camera).</p>
                <div className="flex items-end gap-2 text-xs">
                  <label>rect W <input className="inp w-14" value={rect.w} onChange={(e) => setRect({ ...rect, w: e.target.value })} /></label>
                  <label>D <input className="inp w-14" value={rect.d} onChange={(e) => setRect({ ...rect, d: e.target.value })} /></label>
                  <button className="btn" onClick={fillRect} disabled={pts.length < 4}>fill 1-4 as rectangle</button>
                </div>
                <p className="text-[11px] text-dim">Rectangle order: 1 = near-left, 2 = near-right, 3 = far-right, 4 = far-left.</p>
                <table className="tbl">
                  <thead><tr><th>#</th><th>x (m)</th><th>y (m)</th><th /></tr></thead>
                  <tbody>
                    {world.map((w, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        {[0, 1].map((j) => (
                          <td key={j}>
                            <input className="inp w-16" value={w[j]} onChange={(e) => setWorld((W) => W.map((r, k) => (k === i ? r.map((v, l) => (l === j ? e.target.value : v)) : r)))} />
                          </td>
                        ))}
                        <td><button className="text-dim" onClick={() => { setPts((p) => p.filter((_, k) => k !== i)); setWorld((W) => W.filter((_, k) => k !== i)); }}>✕</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <button className="btn btn-primary" disabled={!worldOk} onClick={() => setStep(3)}>Next</button>
              </>
            )}
            {step === 3 && (
              <>
                <p className="text-dim">Optional: enable approximate height estimation. Choose one:</p>
                {(["none", "camera", "reference"] as const).map((m) => (
                  <label key={m} className="flex items-center gap-2">
                    <input type="radio" checked={heightMode === m} onChange={() => setHeightMode(m)} />
                    {m === "none" ? "No height estimation" : m === "camera" ? "I know the camera height above the floor" : "Mark a reference object of known height"}
                  </label>
                ))}
                {heightMode === "camera" && <label className="block text-xs">camera height (m) <input className="inp w-20" value={camH} onChange={(e) => setCamH(e.target.value)} /></label>}
                {heightMode === "reference" && (
                  <div className="text-xs text-dim">
                    Click the object&apos;s base on the floor, then its top (e.g. a door: 2.0 m).
                    <label className="mt-1 block">height (m) <input className="inp w-20" value={ref.h} onChange={(e) => setRef({ ...ref, h: e.target.value })} /></label>
                    <p>{ref.foot ? "base ✓" : "base ?"} {ref.top ? "top ✓" : "top ?"}</p>
                  </div>
                )}
                <p className="text-[11px] text-dim">Heights use a single-view approximation and are shown with an interval, only when the whole body is visible and upright.</p>
                <button className="btn btn-primary" onClick={doFit}>Compute calibration</button>
                {err && <p className="text-bad">{err}</p>}
              </>
            )}
            {step === 4 && fit && (
              <>
                <p>Status: <b className={fit.status === "calibrated_verified" ? "text-good" : "text-warn"}>{fit.status}</b></p>
                <p className="text-dim">Fit residual: {fit.rms_error_m === null ? "not measurable with 4 points" : `${(fit.rms_error_m * 100).toFixed(1)} cm RMS`}</p>
                <p className="text-dim">Height estimation: {fit.height_estimation ? `enabled (camera ${Number(fit.camera_height_m).toFixed(2)} m, ${fit.camera_height_source})` : "disabled"}</p>
                {(fit.warnings as string[]).map((w) => <p key={w} className="text-[11px] text-warn">⚠ {w}</p>)}
                <p className="text-[11px] text-dim">Check visually: the cyan 1 m grid should lie flat on the floor and the orange horizon should match the scene.</p>
                <div className="flex gap-2">
                  <button className="btn" onClick={() => setStep(2)}>Edit points</button>
                  <button className="btn btn-primary" onClick={() => { onApply(req()); onClose(); }}>Apply</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
