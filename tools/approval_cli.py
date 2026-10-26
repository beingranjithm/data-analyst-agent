"""Turns what a person types at the approval prompt into an ApprovalDecision."""
import re

from agents.approval_workflow import ApprovalDecision

HELP = "Type chart numbers to DROP like 2,3  (Enter = keep everything, q = cancel)."


def parse_answer(text: str) -> ApprovalDecision:
    text = (text or "").strip().lower()
    if not text:
        return ApprovalDecision()
    if text in ("q", "quit", "cancel", "n", "no"):
        return ApprovalDecision(approve=False)
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if not all(p.isdigit() for p in parts):
        raise ValueError(f"I did not understand '{text}'. {HELP}")
    return ApprovalDecision(drop=[int(p) for p in parts])
