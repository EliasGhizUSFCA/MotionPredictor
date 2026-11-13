"""Accelerator selection. Reports what is actually active; never assumes support."""

from __future__ import annotations

import os
import platform

import torch


def select_device(preference: str | None = None) -> dict:
    """preference: 'auto' | 'cuda' | 'mps' | 'cpu' (env MP_DEVICE overrides)."""
    pref = (preference or os.environ.get("MP_DEVICE", "auto")).lower()
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    notes = []
    if pref == "cuda" and not cuda:
        notes.append("CUDA requested but not available; falling back")
    if pref == "mps" and not mps:
        notes.append("MPS requested but not available; falling back")
    if pref in ("auto", "cuda") and cuda:
        dev = "cuda"
    elif pref in ("auto", "mps") and mps:
        dev = "mps"
    else:
        dev = "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else (platform.processor() or platform.machine())
    return {
        "device": dev,
        "device_name": name,
        "cuda_available": cuda,
        "mps_available": mps,
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "notes": notes,
    }
