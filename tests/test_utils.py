import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from folklore_rag.utils import parse_json, score_from, strip_think


def test_parse_json_extracts_object_from_noisy_text():
    assert parse_json('Sure! {"hook": 7, "pacing": 6} hope that helps') == {"hook": 7, "pacing": 6}


def test_parse_json_returns_none_on_garbage():
    assert parse_json("no json here") is None
    assert parse_json("{not valid json}") is None


def test_strip_think_removes_reasoning_block():
    assert strip_think("<think>hmm\nthinking</think>  answer") == "answer"


def test_score_from_averages_only_numeric_rubric_keys():
    data = {"hook": 8, "character_consistency": 6, "pacing": "n/a", "weakest_issue": "x"}
    assert score_from(data) == 7.0


def test_score_from_defaults_when_missing():
    assert score_from(None) == 7.0
    assert score_from({}) == 7.0
    assert score_from({}, default=5.0) == 5.0
