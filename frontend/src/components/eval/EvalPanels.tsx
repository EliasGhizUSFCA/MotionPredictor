"use client";
import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { getJSON, postJSON } from "@/lib/api";
import type { EvalCompare, EvalCompareRun, EvalFlag, EvalPrediction, EvalRunInfo, ModelHorizonSummary } from "@/lib/types";
import { fmt, hk } from "@/lib/viz";
import type { RunData } from "./EvalTab";

const MCOL: Record<string, string> = { baseline: "#94a3b8", learned: "#22d3ee" };
const MNAME: Record<string, string> = { baseline: "A · Kalman baseline", learned: "B · SocialMDN" };
const TT = { background: "#121924", border: "1px solid #273142", fontSize: 12 };

function useKeys(data: RunData) {
  const px = data.summary.coordinate_system === "image_pixels";
  const [norm, setNorm] = useState(px);
  const sfx = px && norm ? "_bh" : "";
  const unit = px ? (norm ? "body heights" : "px") : "m";
  return { px, norm, setNorm, sfx, unit };
}

function UnitToggle({ k }: { k: ReturnType<typeof useKeys> }) {
  if (!k.px) return <span className="text-[11px] text-dim">unit: meters (calibrated ground plane)</span>;
  return (
    <label className="flex items-center gap-1 text-[11px] text-dim">
      <input type="checkbox" checked={k.norm} onChange={(e) => k.setNorm(e.target.checked)} /> normalise pixel errors by body height (compares near and far people)
    </label>
  );
}

function Panel({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-panel p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-dim">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

const stat = (d: ModelHorizonSummary | undefined, key: string) => (d as any)?.[key] as { mean: number; median: number; p90: number } | null | undefined;

// ------------------------------------------------------------------ comparison
export function ComparisonPanel({ data, onJump, onRerun }: { data: RunData; onJump: (id: string, h?: number) => void; onRerun: (id: string) => void }) {
  const k = useKeys(data);
  const { summary } = data;
  const [ckpts, setCkpts] = useState<{ checkpoint: string; active: boolean }[]>([]);
  const [ck, setCk] = useState("");
  const [others, setOthers] = useState<string>("");
  const [cmp, setCmp] = useState<EvalCompare | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    getJSON<{ checkpoints: { checkpoint: string; active: boolean }[] }>("/api/models").then((r) => setCkpts(r.checkpoints)).catch(() => {});
  }, []);

  const byH = summary.horizons.map((h) => {
    const row: Record<string, number | string> = { h: `${h}s` };
    for (const m of summary.models) {
      const d = summary.per_model_horizon[`${m}@${hk(h)}`];
      row[`${m}_ade`] = stat(d, "ade_top1" + k.sfx)?.mean ?? NaN;
      row[`${m}_fde`] = stat(d, "fde_top1" + k.sfx)?.mean ?? NaN;
      row[`${m}_minfde`] = stat(d, "min_fde" + k.sfx)?.mean ?? NaN;
    }
    return row;
  });

  return (
    <div className="space-y-3">
      <Panel title="Accuracy by horizon - identical samples for both models" right={<UnitToggle k={k} />}>
        <div className="grid gap-3 lg:grid-cols-2">
          {(["ade", "fde"] as const).map((m) => (
            <div key={m} className="h-56">
              <p className="text-[11px] text-dim">{m === "ade" ? "ADE" : "FDE"} of the top-ranked path (mean, {k.unit}) - lower is better</p>
              <ResponsiveContainer>
                <BarChart data={byH}>
                  <CartesianGrid stroke="#16202c" />
                  <XAxis dataKey="h" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip contentStyle={TT} formatter={(v: number) => v.toFixed(3)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {summary.models.map((mm) => <Bar key={mm} dataKey={`${mm}_${m}`} name={MNAME[mm]} fill={MCOL[mm]} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
        <div className="mt-3 overflow-auto">
          <table className="tbl">
            <thead>
              <tr>
                <th>model</th><th>horizon</th><th>n</th><th>ADE top-1 mean / median / p90</th><th>FDE top-1 mean / median / p90</th>
                <th>minADE@K / minFDE@K*</th><th>wADE / wFDE†</th><th>miss rate top-1 / @K</th><th>coverage 50% / 90%</th><th>NLL (final)</th><th>latency ms</th><th>probabilities</th>
              </tr>
            </thead>
            <tbody>
              {summary.horizons.flatMap((h) =>
                summary.models.map((m) => {
                  const d = summary.per_model_horizon[`${m}@${hk(h)}`];
                  const s3 = (key: string) => {
                    const s = stat(d, key);
                    return s ? `${fmt(s.mean)} / ${fmt(s.median)} / ${fmt(s.p90)}` : "–";
                  };
                  return (
                    <tr key={`${m}${h}`}>
                      <td style={{ color: MCOL[m] }}>{MNAME[m]}</td>
                      <td>{h}s</td>
                      <td>{d?.n ?? 0}</td>
                      <td className="font-mono">{s3("ade_top1" + k.sfx)}</td>
                      <td className="font-mono">{s3("fde_top1" + k.sfx)}</td>
                      <td className="font-mono">{fmt(stat(d, "min_ade" + k.sfx)?.mean)} / {fmt(stat(d, "min_fde" + k.sfx)?.mean)} (K={d?.k ?? "–"})</td>
                      <td className="font-mono">{fmt(stat(d, "w_ade" + k.sfx)?.mean)} / {fmt(stat(d, "w_fde" + k.sfx)?.mean)}</td>
                      <td className="font-mono">{fmt(d?.miss_rate_top1)} / {fmt(d?.miss_rate_k)}</td>
                      <td className="font-mono">{fmt(d?.coverage50)} / {fmt(d?.coverage90)}</td>
                      <td className="font-mono">{fmt(stat(d, "nll_final")?.median)}</td>
                      <td className="font-mono">{fmt(d?.latency_ms, 2)}</td>
                      <td className={d?.probability_type === "calibrated_probability" ? "text-good" : "text-dim"}>{d?.probability_type ?? "–"}</td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
          <p className="mt-1 text-[11px] text-dim">
            *Best-of-K uses hindsight to pick the closest hypothesis; it measures whether the right path was among the candidates, not whether the model chose it.
            Miss = final error above {k.px ? "1 body height" : "2 m"}. Coverage should match its nominal level (0.50 / 0.90) for well-calibrated uncertainty.
            NLL shown as median of the final-position negative log-likelihood ({summary.coordinate_system}).
            †Probability-weighted ADE / FDE: errors of all hypotheses weighted by the model&apos;s mode weights (learned model only; the weights are
            {summary.models.some((m) => summary.per_model_horizon[`${m}@${hk(summary.selected_horizon_for_rankings)}`]?.probability_type === "calibrated_probability") ? " validated probabilities" : " uncalibrated"} here).
          </p>
        </div>
      </Panel>

      {Object.keys(summary.paired_comparison).length > 0 && (
        <Panel title="Paired difference on identical samples (B − A, bootstrap 95% CI)">
          <table className="tbl">
            <thead><tr><th>horizon</th><th>Δ ADE top-1</th><th>Δ FDE top-1</th><th>pairs</th><th>verdict</th></tr></thead>
            <tbody>
              {Object.entries(summary.paired_comparison).map(([h, v]) => {
                const ci = v.fde_top1.ci95;
                const verdict = ci[1] < 0 ? "B better (CI excludes 0)" : ci[0] > 0 ? "A better (CI excludes 0)" : "no significant difference";
                return (
                  <tr key={h}>
                    <td>{h}s</td>
                    <td className="font-mono">{fmt(v.ade_top1.mean_diff, 3)} [{fmt(v.ade_top1.ci95[0], 3)}, {fmt(v.ade_top1.ci95[1], 3)}] {v.unit}</td>
                    <td className="font-mono">{fmt(v.fde_top1.mean_diff, 3)} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] {v.unit}</td>
                    <td>{v.fde_top1.n_pairs}</td>
                    <td>{verdict}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}

      <Panel title="Compare runs and model versions">
        <div className="flex flex-wrap items-end gap-2 text-xs">
          <label className="text-[11px] text-dim">re-run this evaluation with checkpoint
            <select className="inp block" value={ck} onChange={(e) => setCk(e.target.value)}>
              <option value="">choose…</option>
              {ckpts.map((c) => <option key={c.checkpoint} value={c.checkpoint}>{c.checkpoint}{c.active ? " (active)" : ""}</option>)}
            </select>
          </label>
          <button className="btn" disabled={!ck} onClick={async () => {
            setMsg(null);
            try {
              const r = await postJSON<{ run_id: string }>(`/api/eval/runs/${summary.run_id}/rerun`, { checkpoint: ck });
              onRerun(r.run_id);
            } catch (e: any) {
              setMsg(String(e.message ?? e));
            }
          }}>Re-run</button>
          <label className="ml-4 text-[11px] text-dim">compare with run ids (comma separated)
            <input className="inp block w-72" value={others} onChange={(e) => setOthers(e.target.value)} />
          </label>
          <button className="btn" onClick={async () => {
            try {
              setCmp(await getJSON<EvalCompare>(`/api/eval/compare?runs=${encodeURIComponent([summary.run_id, ...others.split(",").map((s) => s.trim()).filter(Boolean)].join(","))}`));
            } catch (e: any) {
              setMsg(String(e.message ?? e));
            }
          }}>Compare</button>
        </div>
        {msg && <p className="mt-1 text-xs text-bad">{msg}</p>}
        <p className="mt-1 text-[11px] text-dim">Re-runs keep the same source, frames and configuration; only the forecasting checkpoint changes. Note which scenes each checkpoint was trained on - a checkpoint evaluated on its own training scene is not an independent test.</p>
        {cmp && <CompareTables cmp={cmp} />}
      </Panel>
    </div>
  );
}

function CompareTables({ cmp }: { cmp: EvalCompare }) {
  // every distance in a run row is in that row's own unit (m or body heights); rows are grouped by unit and never mixed
  const units = Array.from(new Set(cmp.runs.map((r) => r.unit)));
  const s3 = (v: { mean: number; median: number; p90: number } | null | undefined) => (v ? `${fmt(v.mean)} / ${fmt(v.p90)}` : "–");
  return (
    <div className="mt-2 space-y-2">
      {units.length > 1 && (
        <p className="text-[11px] text-warn">These runs use different units ({units.join(", ")}). Distances are only comparable within one unit group; rows are grouped below.</p>
      )}
      {units.map((u) => (
        <div key={u} className="overflow-auto">
          <p className="text-[11px] text-dim">Unit: <b>{u}</b> (all distances in this table; mean / p90)</p>
          <table className="tbl">
            <thead><tr><th>run</th><th>tags</th><th>learned version</th><th>coords</th><th>model@h</th><th>n</th><th>ADE top-1</th><th>FDE top-1</th><th>minADE*</th><th>minFDE*</th><th>miss</th><th>cov 50 / 90</th><th>ECE</th><th>latency ms</th></tr></thead>
            <tbody>
              {cmp.runs.filter((r) => r.unit === u).flatMap((r) =>
                Object.entries(r.per_model_horizon).map(([key, v]) => (
                  <tr key={r.run_id + key}>
                    <td className="font-mono">{r.run_id}{r.source?.scene_in_active_model_training ? <span className="ml-1 text-[10px] text-warn">in-sample</span> : null}</td>
                    <td>{[r.tags?.sport, r.tags?.camera_perspective].filter(Boolean).join(" · ") || "–"}</td>
                    <td>{r.models?.learned?.version ?? "–"}</td>
                    <td>{r.coordinate_system}</td>
                    <td>{key}</td>
                    <td>{v.n}</td>
                    <td className="font-mono">{s3(v.ade_top1)}</td>
                    <td className="font-mono">{s3(v.fde_top1)}</td>
                    <td className="font-mono">{fmt(v.min_ade?.mean)}</td>
                    <td className="font-mono">{fmt(v.min_fde?.mean)}</td>
                    <td className="font-mono">{fmt(v.miss_rate_top1)}</td>
                    <td className="font-mono">{fmt(v.coverage50)} / {fmt(v.coverage90)}</td>
                    <td className="font-mono">{fmt(v.reliability_ece, 3)}</td>
                    <td className="font-mono">{fmt(v.latency_ms)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      ))}
      <p className="text-[11px] text-dim">*Best-of-K with hindsight. Runs marked in-sample were evaluated on a scene the active learned model was trained on.</p>
      {cmp.paired.length > 0 ? (
        <div className="overflow-auto">
          <p className="text-[11px] text-dim">Paired FDE top-1 difference on identical samples (same source and configuration, matched by track and time; run B − run A, bootstrap 95% CI)</p>
          <table className="tbl">
            <thead><tr><th>run A</th><th>run B</th><th>model</th><th>horizon</th><th>pairs</th><th>Δ FDE top-1</th><th>verdict</th></tr></thead>
            <tbody>
              {cmp.paired.map((p) => {
                const ci = p.fde_top1.ci95;
                return (
                  <tr key={`${p.run_a}${p.run_b}${p.model}${p.horizon}`}>
                    <td className="font-mono">{p.run_a}</td><td className="font-mono">{p.run_b}</td><td>{MNAME[p.model] ?? p.model}</td><td>{p.horizon}s</td><td>{p.n_pairs}</td>
                    <td className="font-mono">{fmt(p.fde_top1.mean_diff, 3)} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] {p.unit ?? ""}</td>
                    <td>{ci[1] < 0 ? "run B better (CI excludes 0)" : ci[0] > 0 ? "run A better (CI excludes 0)" : "no significant difference"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        cmp.runs.length > 1 && <p className="text-[11px] text-dim">No paired comparison: paired differences need runs of the same source (SHA-256 / scene) with the same configuration.</p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------- analytics
function rowsFor(preds: EvalPrediction[], h: number, model: string, sfx: string, flagged: number[]) {
  const out: { t: number; id: string; track: number; ade: number; fde: number; min_fde: number }[] = [];
  const bad = new Set(flagged);
  for (const p of preds) {
    if (bad.has(p.track_id)) continue; // reviewer-flagged inaccurate tracks are excluded from every aggregate (as in summary.json)
    const ev = p.evaluations[hk(h)];
    const s = ev?.included ? ev.scores?.[model] : undefined;
    if (!s) continue;
    out.push({ t: p.t, id: p.id, track: p.track_id, ade: (s as any)["ade_top1" + sfx], fde: (s as any)["fde_top1" + sfx], min_fde: (s as any)["min_fde" + sfx] });
  }
  return out;
}

export function AnalyticsPanel({ data, onJump }: { data: RunData; onJump: (id: string, h?: number) => void }) {
  const k = useKeys(data);
  const { summary, predictions } = data;
  const [h, setH] = useState(summary.selected_horizon_for_rankings);
  const [metricKey, setMetricKey] = useState<"ade" | "fde">("fde");

  const series = useMemo(() => Object.fromEntries(summary.models.map((m) => [m, rowsFor(predictions, h, m, k.sfx, summary.flagged_tracks_excluded)])), [summary.models, predictions, h, k.sfx, summary.flagged_tracks_excluded]);

  const hist = useMemo(() => {
    const all = summary.models.flatMap((m) => series[m].map((r) => r.fde));
    if (!all.length) return [];
    const max = Math.max(...all);
    const nb = 20;
    const bins = Array.from({ length: nb }, (_, i) => ({ x: ((i + 0.5) * max) / nb, ...Object.fromEntries(summary.models.map((m) => [m, 0])) })) as any[];
    for (const m of summary.models) for (const r of series[m]) bins[Math.min(nb - 1, Math.floor((r.fde / (max || 1)) * nb))][m] += 1;
    return bins;
  }, [series, summary.models]);

  const perTrack = useMemo(() => {
    const map = new Map<number, any>();
    for (const m of summary.models)
      for (const r of series[m]) {
        const e = map.get(r.track) ?? { track: `#${r.track}`, n: 0 };
        e[m] = (e[m] ?? 0) + r.fde;
        e[`${m}_n`] = (e[`${m}_n`] ?? 0) + 1;
        map.set(r.track, e);
      }
    return Array.from(map.values()).map((e) => ({ ...e, ...Object.fromEntries(summary.models.map((m) => [m, e[m] / e[`${m}_n`]])) })).slice(0, 40);
  }, [series, summary.models]);

  const learned = summary.per_model_horizon[`learned@${hk(h)}`];
  const rel = learned?.reliability;
  // nearest stored forecast (scored at this horizon, not on a flagged track if possible) to a clicked time
  const jumpNear = (t: number) => {
    if (!Number.isFinite(t) || !predictions.length) return;
    const bad = new Set(summary.flagged_tracks_excluded);
    const pool = predictions.filter((p) => !bad.has(p.track_id) && p.evaluations[hk(h)]?.included);
    const src = pool.length ? pool : predictions;
    let best = src[0];
    for (const p of src) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    onJump(best.id, h);
  };
  const cov = summary.horizons.map((hh) => ({
    h: `${hh}s`,
    ...Object.fromEntries(summary.models.flatMap((m) => [[`${m}_50`, summary.per_model_horizon[`${m}@${hk(hh)}`]?.coverage50 ?? null], [`${m}_90`, summary.per_model_horizon[`${m}@${hk(hh)}`]?.coverage90 ?? null]])),
  }));

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] text-dim">horizon</span>
        {summary.horizons.map((x) => <button key={x} className={`tab ${x === h ? "tab-on" : ""}`} onClick={() => setH(x)}>{x}s</button>)}
        <span className="ml-3 text-[11px] text-dim">metric</span>
        {(["ade", "fde"] as const).map((x) => <button key={x} className={`tab ${x === metricKey ? "tab-on" : ""}`} onClick={() => setMetricKey(x)}>{x.toUpperCase()}</button>)}
        <span className="ml-auto"><UnitToggle k={k} /></span>
      </div>

      <Panel title={`${metricKey.toUpperCase()} over time (top-1 path, ${k.unit}) - click a point to replay it`}>
        <div className="h-64">
          <ResponsiveContainer>
            <ScatterChart margin={{ left: 0, right: 10 }}>
              <CartesianGrid stroke="#16202c" />
              <XAxis dataKey="t" type="number" name="video time" unit="s" stroke="#64748b" fontSize={11} domain={["dataMin", "dataMax"]} />
              <YAxis dataKey={metricKey} type="number" name={metricKey} stroke="#64748b" fontSize={11} />
              <ZAxis range={[18, 18]} />
              <Tooltip contentStyle={TT} formatter={(v: number) => (typeof v === "number" ? v.toFixed(3) : v)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {summary.models.map((m) => (
                <Scatter key={m} name={MNAME[m]} data={series[m]} fill={MCOL[m]} cursor="pointer" onClick={(d: any) => onJump(d.id ?? d.payload?.id, h)} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title={`FDE distribution at ${h}s (${k.unit})`}>
          <div className="h-52">
            <ResponsiveContainer>
              <BarChart data={hist}>
                <CartesianGrid stroke="#16202c" />
                <XAxis dataKey="x" tickFormatter={(v) => v.toFixed(1)} stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} allowDecimals={false} />
                <Tooltip contentStyle={TT} labelFormatter={(v) => `≈ ${Number(v).toFixed(2)}`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {summary.models.map((m) => <Bar key={m} dataKey={m} name={MNAME[m]} fill={MCOL[m]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title={`Error by tracked person at ${h}s (mean FDE, ${k.unit})`}>
          <div className="h-52">
            <ResponsiveContainer>
              <BarChart data={perTrack}>
                <CartesianGrid stroke="#16202c" />
                <XAxis dataKey="track" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={TT} formatter={(v: number) => v.toFixed(3)} />
                {summary.models.map((m) => <Bar key={m} dataKey={m} name={MNAME[m]} fill={MCOL[m]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title={`Mode-probability reliability · SocialMDN @ ${h}s`}>
          {rel?.available ? (
            <>
              <div className="h-52">
                <ResponsiveContainer>
                  <ComposedChart data={rel.bins!.map((b) => ({ c: b.confidence, f: b.frequency, n: b.n }))}>
                    <CartesianGrid stroke="#16202c" />
                    <XAxis dataKey="c" type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} name="predicted probability" />
                    <YAxis type="number" domain={[0, 1]} stroke="#64748b" fontSize={10} />
                    <Tooltip contentStyle={TT} formatter={(v: number) => v.toFixed(3)} />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
                    <Line dataKey="f" stroke="#22d3ee" dot isAnimationActive={false} name="observed frequency" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-dim">
                Event: &ldquo;this path turned out closest to what happened&rdquo;. Points on the diagonal = calibrated. ECE {fmt(rel.ece, 3)} over {rel.n_mode_predictions} mode predictions.
                Status: {learned?.probability_type === "calibrated_probability" ? "probabilities validated offline on held-out data" : "uncalibrated weights in this coordinate system"}.
              </p>
            </>
          ) : (
            <p className="text-xs text-dim">No probabilistic mode outputs at this horizon.</p>
          )}
        </Panel>
        <Panel title="Uncertainty-region coverage vs nominal">
          <div className="h-52">
            <ResponsiveContainer>
              <BarChart data={cov}>
                <CartesianGrid stroke="#16202c" />
                <XAxis dataKey="h" stroke="#64748b" fontSize={10} />
                <YAxis domain={[0, 1]} stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={TT} formatter={(v: number) => v?.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <ReferenceLine y={0.5} stroke="#475569" strokeDasharray="3 3" />
                <ReferenceLine y={0.9} stroke="#475569" strokeDasharray="3 3" />
                {summary.models.flatMap((m) => [
                  <Bar key={`${m}50`} dataKey={`${m}_50`} name={`${MNAME[m]} 50%`} fill={MCOL[m]} fillOpacity={0.5} />,
                  <Bar key={`${m}90`} dataKey={`${m}_90`} name={`${MNAME[m]} 90%`} fill={MCOL[m]} />,
                ])}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-dim">Fraction of actual final positions inside the predicted 50% / 90% region. Below the dashed line = overconfident; above = too cautious.</p>
        </Panel>
      </div>

      <Heatmap data={data} h={h} />

      <TagAccuracy data={data} h={h} unit={k.px ? "body heights" : "m"} />

      <Panel title="Tracking reliability over time">
        <div className="h-48">
          <ResponsiveContainer>
            <LineChart data={summary.tracking_over_time} margin={{ left: -10, right: 10 }} style={{ cursor: "pointer" }} onClick={(e: any) => jumpNear(Number(e?.activeLabel))}>
              <CartesianGrid stroke="#16202c" />
              <XAxis dataKey="t" type="number" unit="s" stroke="#64748b" fontSize={10} domain={["dataMin", "dataMax"]} />
              <YAxis yAxisId="a" stroke="#64748b" fontSize={10} allowDecimals={false} />
              <YAxis yAxisId="b" orientation="right" domain={[0, 1]} stroke="#64748b" fontSize={10} />
              <Tooltip contentStyle={TT} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line yAxisId="a" dataKey="tracks" name="confirmed tracks" stroke="#3b82f6" dot={false} isAnimationActive={false} />
              <Line yAxisId="b" dataKey="mean_tracker_conf" name="mean tracker confidence" stroke="#22c55e" dot={false} isAnimationActive={false} />
              {summary.tracking_over_time.filter((f) => f.cut).map((f) => <ReferenceLine key={f.t} yAxisId="a" x={f.t} stroke="#f43f5e" />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-dim">
          Click the chart to open the stored forecast nearest to that time. Red lines: detected camera cuts (tracks reset; new tracks get new ids). {summary.tracking_over_time.filter((f) => f.camera_moving).length} sampled frames flagged as camera moving.
          Exclusions at {h}s: {Object.entries(summary.inclusion[hk(h)] ?? {}).map(([kk, v]) => `${kk} ${v}`).join(" · ")}.
          {summary.forecast_failures?.count ? ` Forecast failures (eligible tracks with no forecast; no sample for any model): ${summary.forecast_failures.count} - ${Object.entries(summary.forecast_failures.by_reason).map(([kk, v]) => `${kk} ${v}`).join(" · ")}.` : ""}
        </p>
      </Panel>
    </div>
  );
}

// Accuracy grouped by run tag (sport / camera perspective) across all finished runs that share this run's unit.
function TagAccuracy({ data, h, unit }: { data: RunData; h: number; unit: "m" | "body heights" }) {
  const [runs, setRuns] = useState<EvalCompareRun[] | null>(null);
  const [tag, setTag] = useState<"sport" | "camera_perspective">("sport");
  const [err, setErr] = useState<string | null>(null);
  const runId = data.summary.run_id;
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const all = (await getJSON<{ runs: EvalRunInfo[] }>("/api/eval/runs")).runs.filter((r) => r.status === "done").map((r) => r.run_id);
        if (!all.length) return;
        const c = await getJSON<EvalCompare>(`/api/eval/compare?runs=${encodeURIComponent(all.join(","))}`);
        if (live) setRuns(c.runs);
      } catch (e: any) {
        if (live) setErr(String(e.message ?? e));
      }
    })();
    return () => {
      live = false;
    };
  }, [runId, data.summary]);

  const same = (runs ?? []).filter((r) => r.unit === unit);
  const held = same.filter((r) => !r.source?.scene_in_active_model_training);
  const models = data.summary.models;
  const groups = new Map<string, Record<string, any>>();
  for (const r of held) {
    const g = (r.tags?.[tag] || "untagged").trim();
    const e = groups.get(g) ?? { tag: g, runs: 0 };
    e.runs += 1;
    for (const m of models) {
      const v = r.per_model_horizon[`${m}@${hk(h)}`];
      if (!v?.n || !v.fde_top1) continue;
      e[`${m}_s`] = (e[`${m}_s`] ?? 0) + v.fde_top1.mean * v.n;
      e[`${m}_n`] = (e[`${m}_n`] ?? 0) + v.n;
    }
    groups.set(g, e);
  }
  const rows = Array.from(groups.values()).map((e) => ({ ...e, ...Object.fromEntries(models.map((m) => [m, e[`${m}_n`] ? e[`${m}_s`] / e[`${m}_n`] : null])) }));

  return (
    <Panel
      title={`Accuracy by run tag at ${h}s (mean FDE top-1, ${unit})`}
      right={
        <div className="flex gap-1">
          {(["sport", "camera_perspective"] as const).map((t) => (
            <button key={t} className={`tab ${t === tag ? "tab-on" : ""}`} onClick={() => setTag(t)}>{t === "sport" ? "sport / activity" : "camera perspective"}</button>
          ))}
        </div>
      }
    >
      {err && <p className="text-xs text-bad">{err}</p>}
      {runs && rows.length ? (
        <div className="h-52">
          <ResponsiveContainer>
            <BarChart data={rows}>
              <CartesianGrid stroke="#16202c" />
              <XAxis dataKey="tag" stroke="#64748b" fontSize={10} />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip contentStyle={TT} formatter={(v: number) => (typeof v === "number" ? v.toFixed(3) : v)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {models.map((m) => <Bar key={m} dataKey={m} name={MNAME[m]} fill={MCOL[m]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="text-xs text-dim">{runs ? "No finished runs in this unit yet." : "Loading runs…"}</p>
      )}
      <p className="mt-1 text-[11px] text-dim">
        Sample-weighted over {held.length} finished run{held.length === 1 ? "" : "s"} measured in {unit}
        {same.length < (runs?.length ?? 0) ? ` (${(runs?.length ?? 0) - same.length} run(s) in another unit left out - units are never mixed)` : ""}
        {same.length > held.length ? `; ${same.length - held.length} in-sample run(s) (scene in the learned model's training data) left out` : ""}.
        Tag runs when you create them to group them here. By object: every sample is a person (the only tracked class).
        Reviewer-flagged inaccurate tracks are excluded in each run.
      </p>
    </Panel>
  );
}

function Heatmap({ data, h }: { data: RunData; h: number }) {
  const { summary } = data;
  const hm = summary.heatmap;
  const [model, setModel] = useState(summary.models.includes("learned") ? "learned" : summary.models[0]);
  const [layer, setLayer] = useState<"mean_fde" | "exclusion_rate" | "mean_turn_deg">("mean_fde");
  const cells = hm.cells.filter((c) => c.model === model && c.horizon === h);
  const vals = cells.map((c) => (layer === "mean_fde" ? (c.n >= hm.min_samples_to_display ? c.mean_fde : null) : c[layer])).filter((v): v is number => v !== null);
  const max = Math.max(...vals, 1e-6);
  const get = (i: number, j: number) => cells.find((c) => c.i === i && c.j === j);
  return (
    <Panel
      title="Spatial accuracy heatmap (by start position in the image)"
      right={
        <div className="flex flex-wrap gap-1">
          {summary.models.map((m) => <button key={m} className={`tab ${m === model ? "tab-on" : ""}`} onClick={() => setModel(m)}>{MNAME[m]}</button>)}
          {(["mean_fde", "exclusion_rate", "mean_turn_deg"] as const).map((l) => (
            <button key={l} className={`tab ${l === layer ? "tab-on" : ""}`} onClick={() => setLayer(l)}>
              {{ mean_fde: "prediction error", exclusion_rate: "tracking dropouts", mean_turn_deg: "direction changes" }[l]}
            </button>
          ))}
        </div>
      }
    >
      <div className="mx-auto grid max-w-3xl gap-px" style={{ gridTemplateColumns: `repeat(${hm.grid.nx}, minmax(0,1fr))`, aspectRatio: `${hm.grid.x1 - hm.grid.x0}/${hm.grid.y1 - hm.grid.y0}` }}>
        {Array.from({ length: hm.grid.ny }).flatMap((_, j) =>
          Array.from({ length: hm.grid.nx }).map((__, i) => {
            const c = get(i, j);
            const v = c ? (layer === "mean_fde" ? (c.n >= hm.min_samples_to_display ? c.mean_fde : null) : c[layer]) : null;
            const a = v === null ? 0 : v / max;
            return (
              <div key={`${i}-${j}`} className="flex items-center justify-center text-[9px] text-white/80"
                style={{ background: v === null ? "#0f141b" : `rgba(${Math.round(34 + 210 * a)},${Math.round(197 - 140 * a)},${Math.round(94 - 30 * a)},0.85)` }}
                title={c ? `n=${c.n}/${c.n_total} · mean FDE ${fmt(c.mean_fde)} ${hm.unit} · dropouts ${fmt(c.exclusion_rate)} · turns ${fmt(c.mean_turn_deg, 0)}°` : "no samples"}>
                {v !== null ? (layer === "exclusion_rate" ? `${Math.round(v * 100)}%` : fmt(v, layer === "mean_turn_deg" ? 0 : 2)) : ""}
              </div>
            );
          }),
        )}
      </div>
      <p className="mt-1 text-[11px] text-dim">
        Computed from evaluation results at {h}s only. Error cells need ≥ {hm.min_samples_to_display} samples ({hm.unit}). Green = low, red = high.
        &ldquo;Direction changes&rdquo; = mean heading change in the actual future path. Filters: model (buttons) and horizon (above).
        Object type is always person (the only class this system detects and tracks), so there is no object filter. Camera angle and sport are run
        tags: compare them across runs with &ldquo;Accuracy by run tag&rdquo; below or by opening a run with that tag.
      </p>
    </Panel>
  );
}

// ------------------------------------------------------------------- failures
export function FailurePanel({ data, onJump, onChanged }: { data: RunData; onJump: (id: string, h?: number) => void; onChanged: () => void }) {
  const k = useKeys(data);
  const { summary, predictions } = data;
  const [h, setH] = useState(summary.selected_horizon_for_rankings);
  const [model, setModel] = useState(summary.models.includes("learned") ? "learned" : summary.models[0]);
  const [q, setQ] = useState("");
  const [showExcluded, setShowExcluded] = useState(false);
  const [flags, setFlags] = useState<EvalFlag[]>([]);
  useEffect(() => {
    getJSON<any>(`/api/eval/runs/${summary.run_id}/export.json`).then((r) => setFlags(r.flags)).catch(() => {});
  }, [summary.run_id, predictions]);

  const rows = useMemo(() => {
    const bad = new Set(summary.flagged_tracks_excluded);
    const out = predictions.map((p) => {
      const ev = p.evaluations[hk(h)];
      const badTrack = bad.has(p.track_id);
      const s = ev?.included && !badTrack ? ev.scores?.[model] : undefined;
      return { p, ev, s, badTrack, fde: s ? (s as any)["fde_top1" + k.sfx] : null, mfde: s ? (s as any)["min_fde" + k.sfx] : null };
    });
    const term = q.trim().toLowerCase();
    return out
      .filter((r) => (showExcluded ? true : r.s))
      .filter((r) => !term || `#${r.p.track_id} ${r.p.flags.join(" ")} ${r.ev?.exclusion_reasons?.join(" ") ?? ""} ${r.p.t.toFixed(1)}`.toLowerCase().includes(term))
      .sort((a, b) => (b.fde ?? -1) - (a.fde ?? -1))
      .slice(0, 300);
  }, [predictions, h, model, k.sfx, q, showExcluded, summary.flagged_tracks_excluded]);

  const flagged = new Set(flags.map((f) => f.prediction_id).filter(Boolean));
  const unflag = async (p: EvalPrediction) => {
    await postJSON(`/api/eval/runs/${summary.run_id}/flags`, { kind: "unflag", prediction_id: p.id, track_id: null });
    onChanged();
  };
  // undo from the flags table: prediction-level flags by prediction id, track-level 'inaccurate' flags by track id
  const undo = async (f: EvalFlag) => {
    await postJSON(`/api/eval/runs/${summary.run_id}/flags`,
      f.prediction_id ? { kind: "unflag", prediction_id: f.prediction_id, track_id: null } : { kind: "unflag", prediction_id: null, track_id: f.track_id });
    onChanged();
  };

  return (
    <div className="space-y-3">
      <Panel title="High-error predictions" right={<UnitToggle k={k} />}>
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
          <input className="inp w-64" placeholder="search: #track, time, flag, reason…" value={q} onChange={(e) => setQ(e.target.value)} />
          {summary.horizons.map((x) => <button key={x} className={`tab ${x === h ? "tab-on" : ""}`} onClick={() => setH(x)}>{x}s</button>)}
          {summary.models.map((m) => <button key={m} className={`tab ${m === model ? "tab-on" : ""}`} onClick={() => setModel(m)}>{MNAME[m]}</button>)}
          <label className="flex items-center gap-1"><input type="checkbox" checked={showExcluded} onChange={(e) => setShowExcluded(e.target.checked)} /> include excluded samples</label>
        </div>
        <div className="max-h-[520px] overflow-auto">
          <table className="tbl">
            <thead><tr><th>video time</th><th>track</th><th>FDE top-1 ({k.unit})</th><th>best-of-K FDE</th><th>top / closest path</th><th>quality flags</th><th>status</th><th /></tr></thead>
            <tbody>
              {rows.map(({ p, ev, s, badTrack, fde, mfde }) => (
                <tr key={p.id} className="hover:bg-panel2">
                  <td className="font-mono"><button className="underline" onClick={() => onJump(p.id, h)}>{p.t.toFixed(2)} s</button></td>
                  <td>#{p.track_id}</td>
                  <td className="font-mono">{fmt(fde)}</td>
                  <td className="font-mono">{fmt(mfde)}</td>
                  <td>{s ? `${s.top_id} / ${s.best_id}` : "–"}</td>
                  <td className="text-warn">{p.flags.join(", ")}</td>
                  <td>{badTrack ? <span className="text-dim">excluded: track flagged inaccurate</span> : ev?.included ? "scored" : <span className="text-dim">excluded: {ev?.exclusion_reasons?.join(", ")}</span>}</td>
                  <td>{flagged.has(p.id) ? <button className="text-[11px] text-warn underline" onClick={() => unflag(p)}>flagged · undo</button> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel title="Review flags & improvement workflow">
        <ul className="mb-2 list-disc pl-5 text-[11px] text-dim">
          <li>Open a case, inspect the replay, then flag it as a hard case or flag its track as inaccurate (excluded from the statistics after recomputation).</li>
          <li>&ldquo;Review set&rdquo; exports flagged and highest-error cases with only the past observation window and the observed future - usable as fine-tuning data.</li>
          <li>Keep review examples in their own split. Never fine-tune on a recording and report accuracy on that same recording as an independent test.</li>
          <li>After updating a model, use &ldquo;Re-run&rdquo; in Model Comparison to evaluate it on the same source and compare versions.</li>
        </ul>
        <table className="tbl">
          <thead><tr><th>kind</th><th>prediction</th><th>track</th><th>note</th><th>created</th><th /></tr></thead>
          <tbody>
            {flags.map((f, i) => (
              <tr key={i}>
                <td>{f.kind}{f.kind === "annotation_correction" && f.corrected_future ? ` (${f.corrected_future.length} pts)` : ""}</td>
                <td className="font-mono">{f.prediction_id ? <button className="underline" onClick={() => onJump(f.prediction_id!, h)}>{f.prediction_id}</button> : "–"}</td>
                <td>{f.track_id ?? "–"}</td><td>{f.note}</td><td>{f.created}</td>
                <td>{f.prediction_id || f.kind === "inaccurate_track" ? <button className="text-[11px] text-warn underline" onClick={() => undo(f)}>undo</button> : null}</td>
              </tr>
            ))}
            {!flags.length && <tr><td colSpan={6} className="text-dim">No flags yet.</td></tr>}
          </tbody>
        </table>
        {summary.flagged_tracks_excluded.length > 0 && <p className="mt-1 text-[11px] text-warn">Excluded from every statistic and chart as inaccurate tracks: {summary.flagged_tracks_excluded.map((t) => `#${t}`).join(", ")} (use &ldquo;undo&rdquo; above to restore).</p>}
      </Panel>
    </div>
  );
}
