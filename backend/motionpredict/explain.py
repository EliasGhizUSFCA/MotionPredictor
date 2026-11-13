"""Plain-language explanations of pipeline output.

* ``rule_based_summary`` - deterministic text built only from numbers in the frame result.
  Always available; this is the default.
* ``llm_summary`` - optional. Sends the *same structured facts* (no images, no identities) to
  Claude and asks it to restate them. The LLM is never the detector or the predictor, and the
  prompt forbids inventing measurements or calling uncalibrated weights probabilities.
"""

from __future__ import annotations

import json
import os
from typing import Any

LLM_MODEL = os.environ.get("MP_LLM_MODEL", "claude-opus-5-5")

DIRS_IMG = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]


def _dir_name(heading_deg: float, units: str) -> str:
    if units == "px":  # image: y down
        return "toward image " + DIRS_IMG[int(((heading_deg % 360) + 22.5) // 45) % 8]
    return f"heading {heading_deg:.0f} deg in the ground-plane frame"


def facts_for_track(result: dict, track_id: int | None) -> dict[str, Any] | None:
    tracks = [t for t in result.get("tracks", []) if t.get("state") == "confirmed"]
    if track_id is not None:
        tracks = [t for t in tracks if t["track_id"] == track_id]
    if not tracks:
        return None
    t = tracks[0]
    k = t.get("kinematics_world") or t.get("kinematics")
    f = t.get("forecast")
    facts: dict[str, Any] = {
        "track_id": t["track_id"],
        "tracked_for_s": t["age_s"],
        "tracker_confidence": t["tracker_confidence"],
        "coordinate_system": result.get("camera_calibration", {}).get("coordinate_system", "image_pixels"),
        "height": t.get("height"),
        "warnings": result.get("warnings", []),
    }
    if k:
        facts["kinematics"] = {kk: k[kk] for kk in ("units", "speed", "heading_deg", "turn_rate_deg_s", "movement_state", "speed_category")}
    if f:
        facts["forecast"] = {
            "source": f["prediction_source"],
            "model_version": f["model_version"],
            "probability_type": f["probability_type"],
            "calibration_status": f["calibration_status"],
            "horizon_s": f["horizon_seconds"],
            "hypotheses": [
                {"id": tr["id"], "probability": tr["probability"],
                 "end_point": tr["points"][-1], "end_sigma_major": tr["sigma"][-1][0]}
                for tr in f["trajectories"]
            ],
            "residual_probability": f.get("residual_probability"),
            "warnings": f.get("warnings", []),
        }
    return facts


def rule_based_summary(facts: dict) -> str:
    parts = [f"Track {facts['track_id']} has been tracked for {facts['tracked_for_s']:.1f} s "
             f"(tracker confidence {facts['tracker_confidence']:.2f})."]
    k = facts.get("kinematics")
    if k:
        u = "m/s" if k["units"] == "m" else "px/s"
        s = f"It is {k['movement_state']}"
        if k["movement_state"] != "stationary":
            s += f" at {k['speed']:.2f} {u}, {_dir_name(k['heading_deg'], k['units'])}"
            if abs(k["turn_rate_deg_s"]) > 10:
                s += f", turn rate {k['turn_rate_deg_s']:.0f} deg/s"
        if k.get("speed_category"):
            s += f" ({k['speed_category']})"
        parts.append(s + ".")
    h = facts.get("height") or {}
    if h.get("status") == "calibrated":
        parts.append(f"Approximate height {h['meters']:.2f} m (interval {h['interval_m'][0]:.2f}-{h['interval_m'][1]:.2f} m).")
    else:
        parts.append(f"Metric height unknown: {h.get('reason', 'not calibrated')}.")
    f = facts.get("forecast")
    if f:
        if f["source"] == "physics_baseline":
            parts.append(f"The physics baseline extrapolates the current motion for {f['horizon_s']:.1f} s; "
                         f"position uncertainty grows to about {f['hypotheses'][0]['end_sigma_major']:.2f} "
                         f"{'m' if facts['coordinate_system'] == 'ground_plane_meters' else 'px'} (1 sigma). No route probabilities are produced.")
        else:
            top = f["hypotheses"][0]
            label = "probability" if f["probability_type"] == "calibrated_probability" else "uncalibrated weight"
            parts.append(f"The learned model ({f['model_version']}) proposes {len(f['hypotheses'])} paths over {f['horizon_s']:.1f} s; "
                         f"path {top['id']} has the highest {label} ({top['probability']:.0%}).")
            if f["probability_type"] != "calibrated_probability":
                parts.append("These weights are not validated probabilities in the current setting.")
    if facts.get("warnings"):
        parts.append("Warnings: " + " ".join(facts["warnings"]))
    parts.append("Forecasts are estimates, not certain outcomes.")
    return " ".join(parts)


SYSTEM = (
    "You explain the output of a pedestrian tracking and trajectory-forecasting system to a non-expert. "
    "You receive structured facts computed by the system. Restate only what the facts say, in 2-4 sentences. "
    "Never invent numbers, identities, intentions or measurements that are not in the facts. "
    "If probability_type is not 'calibrated_probability', call the values 'uncalibrated model weights', never probabilities or likelihoods. "
    "If the coordinate system is image_pixels, say speeds are in pixels per second and that real-world units are unknown. "
    "Mention relevant warnings. Never describe a forecast as certain. You are not the predictor; you only summarise its output."
)


def llm_summary(facts: dict) -> dict:
    try:
        import anthropic
    except ImportError:
        return {"ok": False, "error": "anthropic package not installed (pip install anthropic)"}
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {"ok": False, "error": "No Anthropic credentials configured (set ANTHROPIC_API_KEY)."}
    try:
        client = anthropic.Anthropic(timeout=30.0)
        resp = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": "Facts (JSON):\n" + json.dumps(facts, default=str)}],
        )
        if resp.stop_reason == "refusal":
            return {"ok": False, "error": "model declined the request"}
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return {"ok": True, "text": text, "model": resp.model}
    except anthropic.RateLimitError:
        return {"ok": False, "error": "rate limited - try again shortly"}
    except anthropic.APIStatusError as e:
        return {"ok": False, "error": f"API error {e.status_code}"}
    except anthropic.APIConnectionError:
        return {"ok": False, "error": "could not reach the Anthropic API"}
