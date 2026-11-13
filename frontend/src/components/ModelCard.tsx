"use client";
import { useEffect, useState } from "react";
import { getJSON } from "@/lib/api";
import { fmt } from "@/lib/viz";

export default function ModelCard() {
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    getJSON<any>("/api/model").then(setD).catch((e) => setErr(String(e.message ?? e)));
  }, []);
  if (err) return <p className="text-bad">{err}</p>;
  if (!d) return <p className="text-dim">loading…</p>;
  const meta = d.forecaster?.meta ?? {};
  const bench = d.benchmark ?? {};
  const scenes = ["eth", "hotel", "univ", "zara1", "zara2"].filter((s) => bench[s]);
  const r = d.report?.model;
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <section className="rounded-lg border border-line bg-panel p-4 text-sm">
        <h2 className="mb-2 text-base text-ink">Active forecaster: {meta.version ?? "none (physics baseline only)"}</h2>
        <ul className="space-y-1 text-xs text-dim">
          <li>Architecture: SocialMDN - GRU encoder over the heading-aligned track, attention over up to 8 neighbours, 5-mode Gaussian-mixture decoder with uncertainty that grows with time.</li>
          <li>Trained on: {(meta.trained_on ?? []).join(", ")} (ETH/UCY pedestrian datasets, world coordinates in meters, 2.5 Hz). Held-out test scene: <b className="text-ink">{meta.held_out_test_scene}</b>.</li>
          <li>Input: up to 8 positions × 0.4 s (2.8 s); output 13 steps (5.2 s). Live tracks are resampled onto this grid.</li>
          <li>Calibration: temperature {fmt(meta.temperature, 3)} on mode scores, σ-scale {fmt(meta.sigma_scale, 3)}; both fitted on a temporal validation split, never on test data.</li>
          <li>
            Probability status:{" "}
            {d.forecaster?.probability_calibration_validated ? (
              <span className="text-good">
                checkpoint gate passed - held-out ECE all modes {fmt(meta.calibration?.test_ece_all_modes, 3)} &lt; {d.forecaster?.ece_threshold}, top mode{" "}
                {fmt(meta.calibration?.test_ece_top_mode, 3)} &lt; {d.forecaster?.ece_top_mode_threshold ?? 0.1}
              </span>
            ) : (
              <span className="text-warn">
                not validated - shown as uncalibrated weights
                {(d.forecaster?.probability_gate?.reasons ?? []).length > 0 && <> ({d.forecaster.probability_gate.reasons.join("; ")})</>}
              </span>
            )}
            . Held-out ECE: all modes {fmt(meta.calibration?.test_ece_all_modes, 3)}, top mode {fmt(meta.calibration?.test_ece_top_mode, 3)}. Even when the
            checkpoint passes, probabilities are only labelled calibrated in ground-plane (meter) mode with a verified camera calibration (or the dataset
            replay); in image-pixel mode weights are always labelled uncalibrated.
          </li>
          <li>Physics baseline: Kalman constant-velocity forecast, noise q = {fmt(meta.kalman_q, 3)} m²/s³ fitted on validation data; reports uncertainty bands, never route probabilities.</li>
        </ul>
        {r && (
          <table className="tbl mt-3">
            <thead><tr><th>held-out {meta.held_out_test_scene}</th><th>ADE</th><th>FDE</th><th>minADE@5</th><th>minFDE@5</th><th>90% coverage @4.8 s</th></tr></thead>
            <tbody>
              <tr><td>SocialMDN (top-1)</td><td>{fmt(r["ade_4.8s"])}</td><td>{fmt(r["fde_4.8s"])}</td><td>{fmt(r["minade_4.8s"])}</td><td>{fmt(r["minfde_4.8s"])}</td><td>{fmt(r.hdr_coverage?.["4.8s"]?.["0.90"])}</td></tr>
              <tr><td>Constant velocity</td><td>{fmt(d.report.baselines.constant_velocity["ade_4.8s"])}</td><td>{fmt(d.report.baselines.constant_velocity["fde_4.8s"])}</td><td>–</td><td>–</td><td>–</td></tr>
              <tr><td>Kalman CV</td><td>{fmt(d.report.baselines.kalman_cv["ade_4.8s"])}</td><td>{fmt(d.report.baselines.kalman_cv["fde_4.8s"])}</td><td>–</td><td>–</td><td>{fmt(d.report.baselines.kalman_cv.hdr_coverage?.["4.8s"]?.["0.90"])}</td></tr>
            </tbody>
          </table>
        )}
      </section>
      <section className="rounded-lg border border-line bg-panel p-4 text-sm">
        <h2 className="mb-2 text-base text-ink">Leave-one-scene-out benchmark (meters, 4.8 s horizon)</h2>
        <table className="tbl">
          <thead><tr><th>test scene</th><th>windows</th><th>CV ADE/FDE</th><th>Kalman ADE/FDE</th><th>MDN top-1</th><th>MDN min-of-5</th><th>mode ECE (all modes)</th><th>90% cov. MDN</th></tr></thead>
          <tbody>
            {scenes.map((s) => {
              const b = bench[s];
              const p = (x: number[]) => `${fmt(x[0])}/${fmt(x[1])}`;
              return (
                <tr key={s}>
                  <td>{s}</td><td>{b.n}</td><td>{p(b["cv_ade_fde_4.8"])}</td><td>{p(b["kalman_ade_fde_4.8"])}</td><td>{p(b["model_top1_ade_fde_4.8"])}</td><td>{p(b["model_min5_ade_fde_4.8"])}</td>
                  <td className={b.ece_cal < 0.05 ? "text-good" : "text-warn"}>{fmt(b.ece_cal, 4)}</td>
                  <td className={b["hdr90_4.8s_model"] >= 0.85 ? "" : "text-warn"}>{fmt(b["hdr90_4.8s_model"], 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] text-dim">
          Honest reading: the learned model&apos;s single most-likely path is about as accurate as constant velocity (a strong, well-known baseline on these datasets);
          its value is in the multiple hypotheses (min-of-5 is ~35% lower) and in better density estimates (lower NLL). Its uncertainty regions are reasonably
          calibrated on ZARA/HOTEL but overconfident on UNIV (crowded) and ETH (out of distribution): see the 90% coverage column. The probability gate needs
          held-out all-modes ECE &lt; 0.05 and top-mode ECE &lt; 0.10; only the zara2 fold passes. ETH (0.102; also top mode 0.168) and UNIV (0.0501) fail on all-modes ECE, HOTEL (0.102)
          and ZARA1 (0.190) fail on top-mode ECE, so their mode values are shown as uncalibrated weights.
        </p>
        <h3 className="mt-3 text-sm text-ink">Limitations</h3>
        <ul className="list-disc pl-5 text-[11px] text-dim">
          <li>Trained on top-down-ish surveillance-style pedestrian data; webcams, sports and crowds with very different dynamics are out of distribution.</li>
          <li>Without camera calibration the model runs on image coordinates normalised by body height - a perspective-distorted approximation, labelled uncalibrated.</li>
          <li>Heights are monocular single-view approximations, available only with calibration and a visible upright full body.</li>
          <li>Not a safety system. Not for surveillance or identification. IDs are temporary and anonymous.</li>
        </ul>
      </section>
    </div>
  );
}
