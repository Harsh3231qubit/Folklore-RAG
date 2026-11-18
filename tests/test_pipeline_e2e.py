"""End-to-end graph test with a fake LLM and retriever (no network, no API key)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("langgraph")

from folklore_rag.config import Settings
from folklore_rag.pipeline import StoryPipeline


class FakeLLM:
    writer = critic = judge = "fake"

    def __init__(self, critic_scores):
        self.critic_scores = list(critic_scores)
        self.writes = 0

    def complete(self, model, system, user, temperature=0.7, max_tokens=700):
        if "Plan an original story" in user:
            return json.dumps({"scenes": ["s1", "s2", "s3", "s4"]})
        if "Rate this scene" in user:
            v = self.critic_scores.pop(0) if self.critic_scores else 9
            return json.dumps({"hook": v, "character_consistency": v, "pacing": v, "emotional_payoff": v,
                               "cultural_authenticity": v, "weakest_issue": "tighten it"})
        self.writes += 1
        return f"scene text {self.writes}"


class FakeRetriever:
    def search(self, key, query, k):
        return [SimpleNamespace(payload={"title": "T", "source": "S", "text": "txt"})]

    def nearest(self, key, text):
        return "T", 0.5


def make(critic_scores):
    llm = FakeLLM(critic_scores)
    return StoryPipeline(Settings(groq_api_key="x"), llm, FakeRetriever()), llm


def test_runs_all_scenes_without_critic():
    pipe, llm = make([])
    out = pipe.run("premise", use_rag=True, use_critic=False)
    assert len(out["scenes"]) == 4 and llm.writes == 4 and out["nearest_title"] == "T"


def test_low_score_triggers_rewrite_then_advances():
    pipe, llm = make([3, 9])  # first scene fails once, then passes
    out = pipe.run("premise", use_rag=False, use_critic=True)
    assert len(out["scenes"]) == 4 and llm.writes == 5 and out["refs"] == []


def test_rewrite_budget_is_capped():
    pipe, llm = make([1] * 50)  # always failing critic
    out = pipe.run("premise", use_critic=True)
    assert len(out["scenes"]) == 4 and llm.writes == 4 * 3  # 1 write + 2 rewrites per scene
