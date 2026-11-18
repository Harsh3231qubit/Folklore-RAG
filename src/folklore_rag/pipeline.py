"""LangGraph story-generation pipeline: retrieve -> plan -> (write -> critic)* -> originality check."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from .config import Settings
from .prompts import SYS_EDITOR, SYS_WRITER, plan_prompt, rubric_prompt, scene_prompt
from .utils import parse_json, score_from

if TYPE_CHECKING:  # heavy imports only needed for type hints
    from .llm import LLMClient
    from .retrieval import Retriever

log = logging.getLogger(__name__)


class State(TypedDict, total=False):
    premise: str
    use_rag: bool
    use_critic: bool
    refs: list[dict]
    outline: list[str]
    idx: int
    current: str
    tries: int
    score: float
    feedback: str
    scenes: list[str]
    scene_scores: list[float]
    story: str
    nearest_title: str
    max_sim: float


def should_rewrite(score: float, tries: int, pass_score: float, max_rewrites: int) -> bool:
    """Rewrite the scene if it scored below threshold and the rewrite budget is not spent."""
    return score < pass_score and tries <= max_rewrites


class StoryPipeline:
    def __init__(self, settings: Settings, llm: LLMClient, retriever: Retriever):
        self.s = settings
        self.llm = llm
        self.retriever = retriever
        self.app = self._build_graph()

    # ---- nodes -----------------------------------------------------------
    def _retrieve(self, st: State) -> dict:
        if not st["use_rag"]:
            return {"refs": []}
        return {"refs": [h.payload for h in self.retriever.search(self.s.embedding, st["premise"], self.s.top_k)]}

    def _plan(self, st: State) -> dict:
        refs = "\n".join(f"- {r['title']}: {r['text']}" for r in st["refs"]) or "(none)"
        raw = self.llm.complete(self.llm.writer, SYS_WRITER, plan_prompt(st["premise"], refs, self.s.n_scenes), 0.7, 600)
        data = parse_json(raw) or {}
        outline = [str(x) for x in data.get("scenes", [])][: self.s.n_scenes]
        if len(outline) < self.s.n_scenes:
            log.warning("Planner returned %d/%d scenes; padding.", len(outline), self.s.n_scenes)
        while len(outline) < self.s.n_scenes:
            outline.append("Continue the story toward a satisfying ending.")
        return {"outline": outline, "idx": 0, "tries": 0, "scenes": [], "scene_scores": [], "feedback": "", "current": ""}

    def _write_scene(self, st: State) -> dict:
        i = st["idx"]
        prev = st["scenes"][-1] if st["scenes"] else "(this is the opening scene)"
        feedback, draft = (st["feedback"], st["current"]) if st["tries"] > 0 else ("", "")
        user = scene_prompt(st["premise"], st["outline"], i, prev, feedback, draft)
        return {"current": self.llm.complete(self.llm.writer, SYS_WRITER, user, 0.8, 500).strip(), "tries": st["tries"] + 1}

    def _critic(self, st: State) -> dict:
        if not st["use_critic"]:
            return {"score": 10.0, "feedback": ""}
        data = parse_json(self.llm.complete(self.llm.critic, SYS_EDITOR, rubric_prompt("scene", st["current"]), 0.0, 400))
        return {"score": score_from(data), "feedback": (data or {}).get("weakest_issue", "")}

    def _route_after_critic(self, st: State) -> str:
        return "write_scene" if should_rewrite(st["score"], st["tries"], self.s.pass_score, self.s.max_rewrites) else "advance"

    def _advance(self, st: State) -> dict:
        return {
            "scenes": st["scenes"] + [st["current"]],
            "scene_scores": st["scene_scores"] + [st["score"]],
            "idx": st["idx"] + 1,
            "tries": 0,
            "feedback": "",
        }

    def _route_after_advance(self, st: State) -> str:
        return "write_scene" if st["idx"] < len(st["outline"]) else "originality_check"

    def _originality(self, st: State) -> dict:
        story = "\n\n".join(st["scenes"])
        title, sim = self.retriever.nearest(self.s.embedding, story)
        return {"story": story, "nearest_title": title, "max_sim": sim}

    # ---- graph -----------------------------------------------------------
    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(State)
        for name, fn in [
            ("retrieve", self._retrieve),
            ("plan", self._plan),
            ("write_scene", self._write_scene),
            ("critic", self._critic),
            ("advance", self._advance),
            ("originality_check", self._originality),
        ]:
            g.add_node(name, fn)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "plan")
        g.add_edge("plan", "write_scene")
        g.add_edge("write_scene", "critic")
        g.add_conditional_edges("critic", self._route_after_critic, {"write_scene": "write_scene", "advance": "advance"})
        g.add_conditional_edges("advance", self._route_after_advance, {"write_scene": "write_scene", "originality_check": "originality_check"})
        g.add_edge("originality_check", END)
        return g.compile()

    def run(self, premise: str, use_rag: bool = True, use_critic: bool = True) -> State:
        return self.app.invoke(
            {"premise": premise, "use_rag": use_rag, "use_critic": use_critic},
            config={"recursion_limit": 100},
        )
