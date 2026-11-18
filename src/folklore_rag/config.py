"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODELS = {
    "e5": "intfloat/multilingual-e5-small",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

RUBRIC = [
    "hook",
    "character_consistency",
    "pacing",
    "emotional_payoff",
    "cultural_authenticity",
]

# Model-id fragments that are not chat models and must be skipped during discovery.
NON_CHAT_MARKERS = ("whisper", "guard", "tts", "orpheus", "compound", "safeguard", "embed", "playai")


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    writer_model: str | None = field(default_factory=lambda: os.environ.get("WRITER_MODEL"))
    critic_model: str | None = field(default_factory=lambda: os.environ.get("CRITIC_MODEL"))
    judge_model: str | None = field(default_factory=lambda: os.environ.get("JUDGE_MODEL"))
    embedding: str = field(default_factory=lambda: os.environ.get("EMBEDDING", "e5"))
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("CACHE_DIR", ROOT / ".cache" / "llm")))
    corpus_path: Path = field(default_factory=lambda: Path(os.environ.get("CORPUS_PATH", ROOT / "data" / "corpus.json")))
    rate_sleep: float = field(default_factory=lambda: float(os.environ.get("RATE_SLEEP", "3.0")))

    # Pipeline knobs
    n_scenes: int = 4
    pass_score: float = 7.0
    max_rewrites: int = 2
    top_k: int = 3

    def validate(self) -> None:
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        if self.embedding not in EMBEDDING_MODELS:
            raise ValueError(f"EMBEDDING must be one of {list(EMBEDDING_MODELS)}")
