"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnalyticsPanel, ComparisonPanel, FailurePanel } from "@/components/eval/EvalPanels";
import ReplayPanel from "@/components/eval/ReplayPanel";
import { API_URL, getJSON, postJSON } from "@/lib/api";
import type { EvalPrediction, EvalRunInfo, EvalSummary } from "@/lib/types";

const H_ALL = [0.5, 1, 2, 3, 5];
const TABS = ["Video Replay", "Model Comparison", "Accuracy Analytics", "Failure Analysis"] as const;
type Tab = (typeof TABS)[number];

async function sha256(file: File): Promise<string | null> {
  if (file.size > 1.5e9 || !crypto?.subtle) return null;
  const d = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export interface RunData {
  info: EvalRunInfo;
  summary: EvalSummary;
  predictions: EvalPrediction[];
  tracks: Record<string, [number, number, number, number | null, number | null][]>;
}

export default function EvalTab() {
  const [runs, setRuns] = useState<EvalRunInfo[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [data, setData] = useState<RunData | null>(null);
  const [tab, setTab] = useState<Tab>("Video Replay");
  const [focus, setFocus] = useState<{ id: string; h?: number } | null>(null);
  const [videos, setVideos] = useState<Record<string, { url: string; match: boolean | null }>>({});
  const [err, setErr] = useState<string | null>(null);
  const [form, setForm] = useState({ interval: 1, horizons: [1, 2, 3, 5] as number[], models: ["baseline", "learned"], fps: 10, sport: "", camera: "", notes: "", keep: false, useCal: false, scene: "zara1", maxS: 120 });
  const fileRef = useRef<HTMLInputElement>(null);
  const attachRef = useRef<HTMLInputElement>(null);
  const pending = useRef<File | null>(null);

  const refresh = useCallback(async () => {
    try {
      setRuns((await getJSON<{ runs: EvalRunInfo[] }>("/api/eval/runs")).runs);
    } catch (e: any) {
      setErr(`backend unreachable: ${e.message}`);
    }
  }, []);
  useEffect(() => {
    refresh();
    const id = setInterval(() => {
      if (runs.some((r) => r.status === "running" || r.status === "created")) refresh();
    }, 1500);
    return () => clearInterval(id);
  }, [refresh, runs]);

  const load = useCallback(async (id: string) => {
    setActive(id);
    setData(null);
    setFocus(null);
    try {
      const [info, summary, preds, tracks] = await Promise.all([
        getJSON<EvalRunInfo>(`/api/eval/runs/${id}`),
        getJSON<EvalSummary>(`/api/eval/runs/${id}/summary`),
        getJSON<{ predictions: EvalPrediction[] }>(`/api/eval/runs/${id}/predictions`),
        getJSON<RunData["tracks"]>(`/api/eval/runs/${id}/tracks`),
      ]);
      setData({ info, summary, predictions: preds.predictions, tracks });
    } catch (e: any) {
      setErr(String(e.message ?? e));
    }
  }, []);

  useEffect(() => {
    const r = runs.find((x) => x.run_id === active);
    if (r?.status === "done" && !data) load(r.run_id);
  }, [runs, active, data, load]);

  const evalConfig = () => {
    let calibration = null;
    if (form.useCal) {
      try {
        calibration = JSON.parse(localStorage.getItem("motionpredict.calibration") ?? "null");
      } catch {}
    }
    return {
      interval_s: form.interval, horizons: form.horizons, models: form.models, process_fps: form.fps, calibration,
      tags: { sport: form.sport || null, camera_perspective: form.camera || null, notes: form.notes || null },
    };
  };

  const upload = async (file: File) => {
    setErr(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("config", JSON.stringify(evalConfig()));
    fd.append("keep_video", String(form.keep));
    try {
      const r = await fetch(`${API_URL}/api/eval/runs`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const { run_id } = await r.json();
      setVideos((v) => ({ ...v, [run_id]: { url: URL.createObjectURL(file), match: true } })); // same browser session: same file
      setActive(run_id);
      setData(null);
      refresh();
    } catch (e: any) {
      setErr(String(e.message ?? e));
    }
  };

  const datasetRun = async () => {
    setErr(null);
    try {
      const { run_id } = await postJSON<{ run_id: string }>("/api/eval/runs/dataset", { scene: form.scene, config: { ...evalConfig(), calibration: null, max_duration_s: form.maxS } });
      setActive(run_id);
      setData(null);
      refresh();
    } catch (e: any) {
      setErr(String(e.message ?? e));
    }
  };

  const attach = async (file: File) => {
    if (!active) return;
    const want = data?.info.meta?.source?.sha256;
    const got = await sha256(file);
    setVideos((v) => ({ ...v, [active]: { url: URL.createObjectURL(file), match: want && got ? want === got : null } }));
  };

  const jump = (id: string, h?: number) => {
    setFocus({ id, h });
    setTab("Video Replay");
  };

  const toggle = <T,>(arr: T[], v: T) => (arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);
  const activeInfo = runs.find((r) => r.run_id === active);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 lg:grid-cols-[360px_minmax(0,1fr)]">
        <section className="space-y-3 rounded-lg border border-line bg-panel p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-dim">New evaluation</h3>
          <p className="text-[11px] text-dim">
            The whole recording is processed in time order. Every forecast is generated and saved using only frames up to its timestamp; later frames are
            used only afterwards, to score it. For uploaded video the &ldquo;actual&rdquo; path is the tracker&apos;s own later observation of the same person
            (pseudo ground truth) - unreliable segments are excluded and reported.
          </p>
          <div className="grid grid-cols-2 gap-2 text-[11px] text-dim">
            <label>forecast every (s)<input className="inp w-full" type="number" min={0.2} step={0.1} value={form.interval} onChange={(e) => setForm({ ...form, interval: Number(e.target.value) })} /></label>
            <label>analysis fps<input className="inp w-full" type="number" min={2} max={30} value={form.fps} onChange={(e) => setForm({ ...form, fps: Number(e.target.value) })} /></label>
            <label>sport / activity (tag)<input className="inp w-full" placeholder="e.g. basketball" value={form.sport} onChange={(e) => setForm({ ...form, sport: e.target.value })} /></label>
            <label>camera perspective (tag)<input className="inp w-full" placeholder="e.g. broadcast side" value={form.camera} onChange={(e) => setForm({ ...form, camera: e.target.value })} /></label>
          </div>
          <div className="flex flex-wrap items-center gap-1 text-xs">
            <span className="text-[11px] text-dim">horizons</span>
            {H_ALL.map((h) => (
              <button key={h} className={`tab ${form.horizons.includes(h) ? "tab-on" : ""}`} onClick={() => setForm({ ...form, horizons: toggle(form.horizons, h).sort((a, b) => a - b) })}>{h}s</button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1 text-xs">
            <span className="text-[11px] text-dim">models</span>
            {["baseline", "learned"].map((m) => (
              <button key={m} className={`tab ${form.models.includes(m) ? "tab-on" : ""}`} onClick={() => setForm({ ...form, models: toggle(form.models, m) })}>{m === "baseline" ? "A: Kalman baseline" : "B: SocialMDN"}</button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={form.useCal} onChange={(e) => setForm({ ...form, useCal: e.target.checked })} /> use the live console&apos;s camera calibration (metric errors)</label>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={form.keep} onChange={(e) => setForm({ ...form, keep: e.target.checked })} /> keep video on the server to allow re-runs (default: deleted after analysis)</label>
          <button className="btn btn-primary w-full" disabled={!form.horizons.length || !form.models.length} onClick={() => fileRef.current?.click()}>Upload video &amp; evaluate…</button>
          <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) { pending.current = f; upload(f); } e.target.value = ""; }} />
          <div className="flex items-center gap-2 border-t border-line pt-2">
            <select className="inp" value={form.scene} onChange={(e) => setForm({ ...form, scene: e.target.value })}>
              {["zara1", "zara2", "eth", "hotel", "univ"].map((x) => <option key={x}>{x}</option>)}
            </select>
            <input className="inp w-16" type="number" value={form.maxS} onChange={(e) => setForm({ ...form, maxS: Number(e.target.value) })} title="seconds" />
            <button className="btn flex-1" onClick={datasetRun}>Evaluate annotated scene</button>
          </div>
          <p className="text-[10px] text-dim">Annotated scenes use human-labelled positions as true ground truth (validates the engine and the models in meters).</p>
          {err && <p className="text-xs text-bad">{err}</p>}
        </section>

        <section className="rounded-lg border border-line bg-panel p-3">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-dim">Evaluation runs</h3>
            <button className="text-[11px] text-dim underline" onClick={refresh}>refresh</button>
          </div>
          <div className="max-h-72 overflow-auto">
            <table className="tbl">
              <thead><tr><th>run</th><th>source</th><th>tags</th><th>status</th><th>forecasts</th><th>coords</th></tr></thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} onClick={() => { setActive(r.run_id); setData(null); if (r.status === "done") load(r.run_id); }} className={`cursor-pointer ${r.run_id === active ? "bg-accent/10" : "hover:bg-panel2"}`}>
                    <td className="font-mono">{r.run_id}</td>
                    <td>{r.meta?.source?.filename ?? (r.request?.scene ? `scene ${r.request.scene}` : "–")}</td>
                    <td>{[r.meta?.tags?.sport, r.meta?.tags?.camera_perspective].filter(Boolean).join(" · ") || "–"}</td>
                    <td className={r.status === "error" ? "text-bad" : r.status === "done" ? "text-good" : "text-warn"}>
                      {r.status}{r.status === "running" ? ` ${(r.progress * 100).toFixed(0)}%` : ""}
                    </td>
                    <td>{r.meta?.n_predictions ?? "–"}</td>
                    <td>{r.meta?.coordinate_system ?? "–"}</td>
                  </tr>
                ))}
                {!runs.length && <tr><td colSpan={6} className="text-dim">No runs yet.</td></tr>}
              </tbody>
            </table>
          </div>
          {activeInfo?.status === "running" && (
            <div className="mt-2 h-1.5 overflow-hidden rounded bg-panel2"><div className="h-full bg-accent transition-all" style={{ width: `${activeInfo.progress * 100}%` }} /></div>
          )}
          {activeInfo?.status === "error" && <p className="mt-2 text-xs text-bad">{activeInfo.error}</p>}
          {data && (
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              <a className="btn" href={`${API_URL}/api/eval/runs/${active}/export.csv`}>CSV</a>
              <a className="btn" href={`${API_URL}/api/eval/runs/${active}/export.json`} download={`motionpredict-${active}.json`}>JSON</a>
              <a className="btn" href={`${API_URL}/api/eval/runs/${active}/report.html`} target="_blank" rel="noreferrer">Report (HTML)</a>
              <a className="btn" href={`${API_URL}/api/eval/runs/${active}/review_set`} download={`review-set-${active}.json`}>Review set</a>
              {data.info.meta?.source?.type === "video" && (
                <>
                  <button className="btn" onClick={() => attachRef.current?.click()}>{videos[active!] ? "Replace local video" : "Attach local video for replay"}</button>
                  <input ref={attachRef} type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && attach(e.target.files[0])} />
                </>
              )}
              {videos[active!]?.match === false && <span className="text-bad">attached file does not match the analysed video (SHA-256)</span>}
              <button className="btn ml-auto text-bad" onClick={async () => { await fetch(`${API_URL}/api/eval/runs/${active}`, { method: "DELETE" }); setActive(null); setData(null); refresh(); }}>Delete run</button>
            </div>
          )}
        </section>
      </div>

      {data && (
        <>
          <div className="flex flex-wrap items-center gap-1">
            {TABS.map((t) => (
              <button key={t} className={`tab px-3 py-1.5 ${tab === t ? "tab-on" : ""}`} onClick={() => setTab(t)}>{t}</button>
            ))}
            <span className="ml-auto text-[11px] text-dim">
              ground truth: {data.summary.ground_truth} · {data.summary.coordinate_system}
              {data.summary.normalised_unit ? " · pixel errors also in body heights" : ""}
            </span>
          </div>
          {tab === "Video Replay" && <ReplayPanel data={data} videoUrl={active ? videos[active]?.url ?? null : null} focus={focus} onFlagged={() => active && load(active)} />}
          {tab === "Model Comparison" && <ComparisonPanel data={data} onJump={jump} onRerun={(id) => { setActive(id); setData(null); refresh(); }} />}
          {tab === "Accuracy Analytics" && <AnalyticsPanel data={data} onJump={jump} />}
          {tab === "Failure Analysis" && <FailurePanel data={data} onJump={jump} onChanged={() => active && load(active)} />}
        </>
      )}
    </div>
  );
}
