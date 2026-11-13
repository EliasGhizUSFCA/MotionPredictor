"""Backend pydantic schemas and frontend/src/lib/types.ts must describe the same fields."""

import re
from pathlib import Path

import pytest

from motionpredict import schemas

TS = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "types.ts").read_text()


def ts_fields(name: str) -> set[str]:
    m = re.search(r"export interface " + name + r" \{(.*?)\n\}", TS, re.S)
    assert m, f"interface {name} missing in types.ts"
    body, depth, top = m.group(1), 0, []
    for line in body.splitlines():
        if depth == 0:
            f = re.match(r"\s*([a-z_0-9]+)\??:", line)
            if f:
                top.append(f.group(1))
        depth += line.count("{") - line.count("}")
    return set(top)


@pytest.mark.parametrize("model,iface", [
    (schemas.StageEvent, "StageEvent"), (schemas.HeightEstimate, "HeightEstimate"), (schemas.Kinematics, "Kinematics"),
    (schemas.Trajectory, "Trajectory"), (schemas.Forecast, "Forecast"), (schemas.TrackOut, "TrackOut"),
    (schemas.FrameResult, "FrameResult"), (schemas.PipelineConfig, "PipelineConfig"), (schemas.CalibrationRequest, "CalibrationRequest"),
])
def test_fields_match(model, iface):
    assert set(model.model_fields) == ts_fields(iface)


def test_default_config_matches():
    m = re.search(r"DEFAULT_CONFIG: PipelineConfig = \{(.*?)\};", TS, re.S)
    ts = dict(re.findall(r"([a-z_]+): ([^,\n]+)", m.group(1)))
    py = schemas.PipelineConfig().model_dump()
    assert set(ts) == set(py)
    for k, v in py.items():
        t = ts[k].strip().strip('"')
        assert (t == str(v).lower()) if isinstance(v, bool) else (str(v) == t or float(t) == float(v)), k
