"use client";
// Capture -> WebSocket -> results. Capture, network and rendering are decoupled:
// at most MAX_IN_FLIGHT frames are outstanding, so the server always works on recent frames
// and stale frames are skipped (counted as client-side drops) instead of queueing up.

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "./api";
import type { CalibrationRequest, FrameResult, PipelineConfig } from "./types";

export type SourceKind = "webcam" | "file" | "demo" | "server";
export type Status = "idle" | "connecting" | "streaming" | "paused" | "error";

const MAX_IN_FLIGHT = 2;

export interface StreamMetrics {
  cameraFps: number | null;
  sentFps: number | null;
  resultFps: number | null;
  latencyMs: number | null;
  clientDropped: number;
  serverDropped: number;
  resolution: string;
}

export interface StreamOptions {
  sendWidth: number;
  maxSendFps: number;
  jpegQuality: number;
  camWidth: number;
  camHeight: number;
}

export const DEFAULT_STREAM_OPTS: StreamOptions = { sendWidth: 640, maxSendFps: 15, jpegQuality: 0.7, camWidth: 1280, camHeight: 720 };

function ema(prev: number | null, v: number, a = 0.2) {
  return prev === null ? v : prev * (1 - a) + v * a;
}

export function useStream(config: PipelineConfig, opts: StreamOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [source, setSource] = useState<SourceKind | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FrameResult | null>(null);
  const [demoFrame, setDemoFrame] = useState<string | null>(null);
  const [calibrationAck, setCalibrationAck] = useState<Record<string, any> | null>(null);
  const [metrics, setMetrics] = useState<StreamMetrics>({ cameraFps: null, sentFps: null, resultFps: null, latencyMs: null, clientDropped: 0, serverDropped: 0, resolution: "–" });
  const [demoInfo, setDemoInfo] = useState<Record<string, any> | null>(null);
  const historyRef = useRef<FrameResult[]>([]);
  const st = useRef({
    inFlight: 0, frameId: 0, lastSend: 0, lastResult: 0, lastCamFrame: 0, sentAt: new Map<number, number>(),
    loop: 0, paused: false, wanted: false, retries: 0, kind: null as SourceKind | null, fileUrl: null as string | null,
    serverUrl: "", demoScene: "zara1", clientDropped: 0, configRef: config, calib: null as CalibrationRequest | null,
  });
  st.current.configRef = config;

  const sendJSON = useCallback((m: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m));
  }, []);

  // push config changes to the server
  useEffect(() => {
    sendJSON({ type: "config", ...config });
  }, [config, sendJSON]);

  const stopCapture = useCallback(() => {
    cancelAnimationFrame(st.current.loop);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const handleMessage = useCallback((ev: MessageEvent) => {
    const msg = JSON.parse(ev.data as string);
    const s = st.current;
    if (msg.type === "result") {
      const r = msg as FrameResult;
      s.inFlight = Math.max(0, s.inFlight - 1);
      const now = performance.now();
      const sent = s.sentAt.get(r.frame_id);
      s.sentAt.delete(r.frame_id);
      // compute intervals now: React runs the updater later, after lastResult is overwritten
      const gap = s.lastResult ? now - s.lastResult : null;
      const lat = sent !== undefined ? now - sent : r.source !== "browser" ? r.processing_ms : null;
      setMetrics((m) => ({
        ...m,
        latencyMs: lat !== null ? ema(m.latencyMs, lat) : m.latencyMs,
        resultFps: gap !== null ? ema(m.resultFps, 1000 / Math.max(gap, 1)) : m.resultFps,
        serverDropped: r.stats.frames_dropped,
        clientDropped: s.clientDropped,
        resolution: `${r.frame_width}×${r.frame_height}`,
      }));
      s.lastResult = now;
      if (r.frame_jpeg_b64) setDemoFrame(`data:image/jpeg;base64,${r.frame_jpeg_b64}`);
      (r as any)._receivedAt = now;
      historyRef.current.push(r);
      if (historyRef.current.length > 600) historyRef.current.splice(0, historyRef.current.length - 600);
      setResult(r);
    } else if (msg.type === "hello") {
      setStatus(s.paused ? "paused" : "streaming");
      s.retries = 0;
      if (msg.demo) setDemoInfo(msg.demo);
      wsRef.current?.send(JSON.stringify({ type: "config", ...s.configRef }));
      if (s.calib && s.kind !== "demo") wsRef.current?.send(JSON.stringify({ type: "calibration", ...s.calib }));
      if (s.kind === "server") wsRef.current?.send(JSON.stringify({ type: "server_source", url: s.serverUrl }));
    } else if (msg.type === "ack" && msg.what === "calibration") {
      setCalibrationAck(msg.calibration);
    } else if (msg.type === "ack" && msg.what === "clear_calibration") {
      setCalibrationAck(null);
    } else if (msg.type === "error") {
      setError(`${msg.where}: ${typeof msg.error === "string" ? msg.error : JSON.stringify(msg.error)}`);
      if (msg.where === "pipeline" || msg.where === "decode") st.current.inFlight = Math.max(0, st.current.inFlight - 1);
    }
  }, []);

  const connect = useCallback(
    (path: string) => {
      const ws = new WebSocket(`${WS_URL}${path}`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      st.current.inFlight = 0;
      ws.onmessage = handleMessage;
      ws.onerror = () => setError("connection error - is the backend running?");
      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        const s = st.current;
        if (s.wanted && s.retries < 20) {
          // graceful reconnect with backoff
          s.retries += 1;
          setStatus("connecting");
          setTimeout(() => s.wanted && wsRef.current === ws && connect(path), Math.min(500 * 2 ** s.retries, 8000));
        } else if (s.wanted) {
          setStatus("error");
        }
      };
    },
    [handleMessage],
  );

  const captureLoop = useCallback(() => {
    const s = st.current;
    const tick = () => {
      s.loop = requestAnimationFrame(tick);
      const v = videoRef.current;
      const ws = wsRef.current;
      if (!v || !ws || ws.readyState !== WebSocket.OPEN || s.paused || v.readyState < 2 || v.videoWidth === 0) return;
      const now = performance.now();
      if (now - s.lastSend < 1000 / opts.maxSendFps) return;
      if (s.inFlight >= MAX_IN_FLIGHT) {
        s.clientDropped += 1; // frame available but server busy: skip it rather than queue it
        return;
      }
      s.lastSend = now;
      const c = (canvasRef.current ??= document.createElement("canvas"));
      const w = Math.min(opts.sendWidth, v.videoWidth);
      const h = Math.round((v.videoHeight / v.videoWidth) * w);
      c.width = w;
      c.height = h;
      c.getContext("2d")!.drawImage(v, 0, 0, w, h);
      const fid = ++s.frameId;
      // media time for files (true recording timestamps), wall clock for live cameras
      const captureTs = s.kind === "file" ? v.currentTime * 1000 : Date.now();
      s.inFlight += 1;
      s.sentAt.set(fid, now);
      c.toBlob(
        async (blob) => {
          if (!blob || ws.readyState !== WebSocket.OPEN) {
            s.inFlight = Math.max(0, s.inFlight - 1);
            return;
          }
          const header = new TextEncoder().encode(JSON.stringify({ frame_id: fid, client_ts_ms: Date.now(), capture_ts_ms: captureTs }));
          const buf = new Uint8Array(4 + header.length + blob.size);
          new DataView(buf.buffer).setUint32(0, header.length);
          buf.set(header, 4);
          buf.set(new Uint8Array(await blob.arrayBuffer()), 4 + header.length);
          ws.send(buf);
        },
        "image/jpeg",
        opts.jpegQuality,
      );
    };
    s.loop = requestAnimationFrame(tick);
    // measure actual camera/video frame rate where supported
    const v = videoRef.current as any;
    if (v?.requestVideoFrameCallback) {
      const cb = () => {
        const n = performance.now();
        const gap = s.lastCamFrame ? n - s.lastCamFrame : null;
        if (gap !== null) setMetrics((m) => ({ ...m, cameraFps: ema(m.cameraFps, 1000 / Math.max(gap, 1), 0.05) }));
        s.lastCamFrame = n;
        if (s.wanted) v.requestVideoFrameCallback(cb);
      };
      v.requestVideoFrameCallback(cb);
    }
  }, [opts.maxSendFps, opts.sendWidth, opts.jpegQuality]);

  const start = useCallback(
    async (kind: SourceKind, arg?: File | string) => {
      const s = st.current;
      stop();
      setError(null);
      setResult(null);
      setDemoFrame(null);
      setDemoInfo(null);
      historyRef.current = [];
      s.kind = kind;
      s.wanted = true;
      s.paused = false;
      s.clientDropped = 0;
      s.lastCamFrame = 0;
      setSource(kind);
      setStatus("connecting");
      try {
        if (kind === "webcam") {
          const ms = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: opts.camWidth }, height: { ideal: opts.camHeight } }, audio: false });
          streamRef.current = ms;
          ms.getVideoTracks()[0].onended = () => {
            setError("camera disconnected - press Start to reconnect");
            setStatus("error");
          };
          const v = videoRef.current!;
          v.srcObject = ms;
          await v.play();
          connect("/ws/stream");
          captureLoop();
        } else if (kind === "file") {
          const v = videoRef.current!;
          if (s.fileUrl) URL.revokeObjectURL(s.fileUrl);
          s.fileUrl = URL.createObjectURL(arg as File);
          v.srcObject = null;
          v.src = s.fileUrl;
          v.loop = true;
          await v.play();
          connect("/ws/stream");
          captureLoop();
        } else if (kind === "demo") {
          s.demoScene = (arg as string) || "zara1";
          connect(`/ws/demo?scene=${encodeURIComponent(s.demoScene)}`);
        } else if (kind === "server") {
          s.serverUrl = arg as string;
          connect("/ws/stream");
        }
      } catch (e: any) {
        setError(e?.name === "NotAllowedError" ? "camera permission denied" : String(e?.message ?? e));
        setStatus("error");
        s.wanted = false;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [connect, captureLoop, opts.camWidth, opts.camHeight],
  );

  const stop = useCallback(() => {
    const s = st.current;
    s.wanted = false;
    stopCapture();
    const ws = wsRef.current;
    wsRef.current = null;
    ws?.close(); // server clears all per-person buffers when the session closes
    const v = videoRef.current;
    if (v) {
      v.pause();
      v.srcObject = null;
    }
    setStatus("idle");
    setSource(null);
  }, [stopCapture]);

  const setPaused = useCallback(
    (p: boolean) => {
      st.current.paused = p;
      if (st.current.kind === "demo") sendJSON({ type: "pause", paused: p });
      if (st.current.kind === "file") (p ? videoRef.current?.pause() : videoRef.current?.play());
      setStatus((x) => (x === "streaming" || x === "paused" ? (p ? "paused" : "streaming") : x));
    },
    [sendJSON],
  );

  const setCalibration = useCallback(
    (c: CalibrationRequest | null) => {
      st.current.calib = c;
      sendJSON(c ? { type: "calibration", ...c } : { type: "clear_calibration" });
      if (!c) setCalibrationAck(null);
    },
    [sendJSON],
  );

  const reset = useCallback(() => {
    sendJSON({ type: "reset" });
    historyRef.current = [];
  }, [sendJSON]);

  useEffect(() => () => stop(), [stop]);

  return {
    videoRef, status, source, error, result, metrics, demoFrame, demoInfo, calibrationAck, history: historyRef,
    start, stop, setPaused, setCalibration, reset, sendJSON,
  };
}
