"""Retrieval and LLM-as-judge evaluations."""
from __future__ import annotations

from .config import RUBRIC
from .llm import LLMClient
from .prompts import SYS_EDITOR, rubric_prompt
from .retrieval import Retriever
from .utils import parse_json

# (query, expected title). Mix of English and Hindi paraphrases of corpus stories.
RETRIEVAL_EVAL = [
    ("a clever rabbit tricks a tyrant lion into jumping into a well", "The Lion and the Hare"),
    ("एक चालाक खरगोश ने शेर को कुएँ में कूदने पर मजबूर किया", "The Lion and the Hare"),
    ("a monkey escapes a crocodile by saying his heart is left on the tree", "The Monkey and the Crocodile"),
    ("a jackal dyed blue is crowned king but gives himself away by howling", "The Blue Jackal"),
    ("a man kills his loyal pet in haste and later learns the truth", "The Brahmin and the Mongoose"),
    ("a poor boy at a fair buys tongs for his grandmother instead of toys", "Idgah"),
    ("emperor refuses reward to a man who stood in cold water, and a witty minister proves a point with cooking", "Birbal's Khichdi"),
    ("a greedy wife plucks every feather of a magical bird", "The Golden Goose"),
    ("a bird talks and falls from the sky while being carried by two friends", "The Talkative Tortoise"),
    ("a king sacrifices his body to make a bridge for his troop", "The Monkey King"),
]


def recall_at_k(retriever: Retriever, key: str, k: int = 3) -> float:
    hits = sum(any(h.payload["title"] == expected for h in retriever.search(key, q, k)) for q, expected in RETRIEVAL_EVAL)
    return hits / len(RETRIEVAL_EVAL)


def judge_story(llm: LLMClient, story: str) -> dict:
    """Score a finished story with the judge model on the shared rubric."""
    raw = llm.complete(llm.judge, SYS_EDITOR, rubric_prompt("complete short story", story, "Judge the story as a whole."), 0.0, 1500)
    data = parse_json(raw)
    res = {k: (float(data[k]) if data and isinstance(data.get(k), (int, float)) else None) for k in RUBRIC}
    vals = [v for v in res.values() if v is not None]
    res["judge_overall"] = sum(vals) / len(vals) if vals else None
    res["judge_note"] = (data or {}).get("weakest_issue", "")
    return res
