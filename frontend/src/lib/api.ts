// REST helpers. The backend is local by default; nothing is sent to third parties from here.

export const API_URL: string =
  process.env.NEXT_PUBLIC_API_URL ?? (typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:8000");

export const WS_URL = API_URL.replace(/^http/, "ws");

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_URL}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = await r.text();
    try {
      msg = JSON.parse(msg).detail ?? msg;
    } catch {}
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export function download(name: string, data: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
