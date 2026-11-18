import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from folklore_rag.corpus import load_corpus
from folklore_rag.pipeline import should_rewrite
from folklore_rag.prompts import scene_prompt

OUTLINE = ["a", "b", "c", "d"]


def test_final_scene_forbids_cliffhanger():
    last = scene_prompt("p", OUTLINE, 3, "prev")
    assert "FINAL scene" in last and "Do NOT end on a cliffhanger" in last


def test_middle_scene_pulls_listener_forward():
    mid = scene_prompt("p", OUTLINE, 1, "prev")
    assert "pulls the listener into the next scene" in mid and "FINAL" not in mid


def test_revision_block_only_when_feedback_given():
    assert "reviewer" not in scene_prompt("p", OUTLINE, 0, "prev")
    assert "reviewer" in scene_prompt("p", OUTLINE, 0, "prev", feedback="fix pacing", draft="old")


def test_rewrite_routing():
    assert should_rewrite(5.0, 1, 7.0, 2) is True      # low score, budget left
    assert should_rewrite(5.0, 3, 7.0, 2) is False     # budget spent
    assert should_rewrite(8.0, 1, 7.0, 2) is False     # good enough


def test_corpus_is_valid():
    corpus = load_corpus(Path(__file__).resolve().parents[1] / "data" / "corpus.json")
    assert len(corpus) >= 12 and len({s["title"] for s in corpus}) == len(corpus)
