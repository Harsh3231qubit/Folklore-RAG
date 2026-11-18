"""Small, dependency-free helpers (kept separate so they are easy to unit test)."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import RUBRIC


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a model response, or None if invalid."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by some reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def score_from(data: dict[str, Any] | None, default: float = 7.0) -> float:
    """Average the rubric scores in a critic response.

    Falls back to ``default`` when the response is missing or has no numeric
    scores, so a malformed critic reply never blocks the pipeline.
    """
    if not data:
        return default
    vals = [float(data[k]) for k in RUBRIC if isinstance(data.get(k), (int, float))]
    return sum(vals) / len(vals) if vals else default
