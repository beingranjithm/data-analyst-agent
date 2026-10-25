"""Sanity checks on a finished Plotly figure (is it empty? all missing? all zero? untitled?)."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FigureCheck:
    problems: list = field(default_factory=list)    # blocking: the chart should not be shown
    warnings: list = field(default_factory=list)    # shown with the chart, not blocking
    missing_title: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def decode(v) -> Optional[np.ndarray]:
    """Plotly 6+/7 may store numbers as {'dtype': 'f8', 'bdata': '...'} after a JSON round trip."""
    if v is None:
        return None
    try:
        if isinstance(v, dict):
            if "bdata" not in v:
                return None
            arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
            if v.get("shape"):
                arr = arr.reshape([int(s) for s in str(v["shape"]).split(",")])
            return arr
        return np.asarray(v)
    except Exception:
        return None


def _numeric(arr: np.ndarray) -> Optional[np.ndarray]:
    try:
        return arr.astype(float).ravel()
    except (ValueError, TypeError):
        return None


def check_figure(fig) -> FigureCheck:
    out = FigureCheck()
    out.missing_title = not (fig.layout.title and fig.layout.title.text)
    traces = list(fig.data)
    if not traces:
        out.problems.append("the figure has no data traces")
        return out

    points, numeric = 0, []
    for t in traces:
        for key in ("x", "y", "z", "values"):
            arr = decode(getattr(t, key, None))
            if arr is None or arr.size == 0:
                continue
            points = max(points, int(arr.shape[0]) if key != "z" else int(arr.size))
            if key != "x" or t.type == "histogram" or getattr(t, "orientation", None) == "h":
                num = _numeric(arr)
                if num is not None:
                    numeric.append(num)
    if points == 0:
        out.problems.append("the chart contains no data points")
        return out
    if numeric:
        finite = np.concatenate([n[np.isfinite(n)] for n in numeric])
        if finite.size == 0:
            out.problems.append("every value in the chart is missing (NaN)")
        elif not np.any(finite != 0):
            out.warnings.append("every value in the chart is zero, so the chart may be wrong")
    if points == 1 and not out.problems:
        out.warnings.append("the chart shows only one data point")
    return out
