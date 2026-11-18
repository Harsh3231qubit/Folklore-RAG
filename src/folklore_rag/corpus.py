"""Corpus loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class Story(TypedDict):
    title: str
    source: str
    text: str


def load_corpus(path: str | Path) -> list[Story]:
    stories = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, s in enumerate(stories):
        missing = {"title", "source", "text"} - s.keys()
        if missing:
            raise ValueError(f"Story #{i} is missing fields: {sorted(missing)}")
    return stories
