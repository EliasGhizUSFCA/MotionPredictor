import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MP_SKIP_MODEL_LOAD", "1")
RAW = ROOT / "data" / "raw"
HAS_DATA = (RAW / "crowds_zara01.txt").exists()
