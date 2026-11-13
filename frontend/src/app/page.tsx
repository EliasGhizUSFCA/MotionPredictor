"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import ModelCard from "@/components/ModelCard";
import { getJSON } from "@/lib/api";
import type { Health } from "@/lib/types";

const LiveConsole = dynamic(() => import("@/components/live/LiveConsole"), { ssr: false });
const EvalTab = dynamic(() => import("@/components/eval/EvalTab"), { ssr: false });

const TABS = ["Live Console", "Prediction Accuracy", "Model & Data"] as const;

export default function Page() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Live Console");
  const [health, setHealth] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const load = () => getJSON<Health>("/api/health").then((h) => { setHealth(h); setErr(null); }).catch((e) => setErr(String(e.message ?? e)));
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);
  return (
    <main className="mx-auto max-w-[1800px] space-y-3 p-3 md:p-4">
      <header className="flex flex-wrap items-center gap-3 border-b border-line pb-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            MotionPredict <span className="text-accent">AI</span>
          </h1>
          <p className="text-[11px] text-dim">Pedestrian tracking & probabilistic trajectory forecasting · research prototype, not for safety-critical or surveillance use</p>
        </div>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button key={t} onClick={() => setTab(t)} className={`tab px-3 py-1.5 text-xs ${tab === t ? "tab-on" : ""}`}>{t}</button>
          ))}
        </nav>
        <div className="ml-auto flex flex-wrap gap-2 text-[11px] text-dim">
          {health ? (
            <>
              <span className="chip">device: {health.device.device} ({health.device.device_name || "?"})</span>
              <span className={`chip ${health.detector.loaded ? "" : "text-bad"}`}>detector: {health.detector.loaded ? health.detector.weights : "not loaded"}</span>
              <span className={`chip ${health.forecaster?.available ? "" : "text-warn"}`}>forecaster: {health.forecaster?.version ?? "baseline only"}</span>
              <span className="chip">local processing · no recording</span>
            </>
          ) : (
            <span className="chip text-bad">backend offline {err ? `(${err})` : ""}</span>
          )}
        </div>
      </header>
      <div hidden={tab !== "Live Console"}><LiveConsole health={health} /></div>
      {tab === "Prediction Accuracy" && <EvalTab />}
      {tab === "Model & Data" && <ModelCard />}
    </main>
  );
}
