"""Pydantic schema for the dashboard plan the planner agent returns.

The plan only references columns by NAME. validate_plan_against_profile() checks every
name and role against the real profile, so a hallucinated column can never reach a chart.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AGGS = ("sum", "mean", "median", "min", "max", "count")
Agg = Literal["sum", "mean", "median", "min", "max", "count"]


class KpiSpec(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    measure: str
    agg: Agg = "sum"


class ChartSpec(BaseModel):
    """One chart. Which fields are required depends on `type` (checked below)."""
    type: Literal["line", "bar", "donut", "histogram", "box", "scatter", "heatmap"]
    title: str = Field(min_length=1, max_length=90)
    why: str = Field(default="", max_length=300)
    time: Optional[str] = None
    dimension: Optional[str] = None
    measure: Optional[str] = None
    agg: Optional[Agg] = None
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    measures: Optional[list[str]] = None

    def to_spec(self) -> dict:
        """The plain dict that tools.charts.build_chart expects (no title/why)."""
        keep = {"line": ["time", "measure", "agg"], "bar": ["dimension", "measure", "agg"],
                "donut": ["dimension", "measure", "agg"], "histogram": ["measure"],
                "box": ["dimension", "measure"], "scatter": ["x", "y", "color"],
                "heatmap": ["measures"]}[self.type]
        out = {"type": self.type}
        for k in keep:
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


REQUIRED = {"line": ["time", "measure"], "bar": ["dimension", "measure"],
            "donut": ["dimension", "measure"], "histogram": ["measure"],
            "box": ["dimension", "measure"], "scatter": ["x", "y"], "heatmap": ["measures"]}


class DashboardPlan(BaseModel):
    title: str = Field(min_length=1, max_length=90)
    summary: str = Field(default="", max_length=500)
    kpis: list[KpiSpec] = Field(default_factory=list, max_length=6)
    charts: list[ChartSpec] = Field(min_length=1, max_length=10)

    @field_validator("charts")
    @classmethod
    def _required_fields(cls, charts):
        for i, c in enumerate(charts, 1):
            missing = [f for f in REQUIRED[c.type] if not getattr(c, f)]
            if missing:
                raise ValueError(f"chart {i} ({c.type}) is missing: {', '.join(missing)}")
            if c.type == "heatmap" and len(c.measures) < 2:
                raise ValueError(f"chart {i} (heatmap) needs at least 2 measures")
        return charts


def validate_plan_against_profile(plan: DashboardPlan, profile: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = plan is usable)."""
    roles = {c["name"]: c["role"] for c in profile["columns"]}
    problems: list[str] = []

    def need(where: str, col: Optional[str], allowed: tuple[str, ...]):
        if col is None:
            return
        if col not in roles:
            problems.append(f"{where}: column '{col}' does not exist")
        elif roles[col] not in allowed:
            problems.append(f"{where}: column '{col}' has role '{roles[col]}', expected {'/'.join(allowed)}")

    for k in plan.kpis:
        need(f"KPI '{k.label}'", k.measure, ("numeric",) if k.agg != "count" else tuple(set(roles.values()) - {"empty"}))
    for i, c in enumerate(plan.charts, 1):
        w = f"chart {i} ({c.type})"
        need(w, c.time, ("datetime",))
        need(w, c.dimension, ("categorical", "boolean"))
        need(w, c.measure, ("numeric",))
        need(w, c.x, ("numeric",))
        need(w, c.y, ("numeric",))
        need(w, c.color, ("categorical", "boolean"))
        for m in c.measures or []:
            need(w, m, ("numeric",))
    return problems
