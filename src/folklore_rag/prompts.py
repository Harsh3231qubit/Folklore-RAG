"""All prompt text lives here so it can be iterated on without touching pipeline logic."""
from __future__ import annotations

from .config import RUBRIC

SYS_WRITER = (
    "You are a scriptwriter for Indian audio dramas and short story series. You write vivid, dialogue-driven scenes "
    "with distinct character voices, concrete sensory detail, emotional stakes and culturally grounded settings. "
    "Avoid cliches and generic morals."
)

SYS_EDITOR = (
    "You are a strict, experienced story editor for an audio-drama platform. Be critical and specific. "
    "Do not inflate scores."
)


def plan_prompt(premise: str, refs_text: str, n_scenes: int) -> str:
    return (
        f"Premise: {premise}\n\nReference folktales (use ONLY for structure and tone, do NOT copy their plots):\n{refs_text}\n\n"
        f"Plan an original story in exactly {n_scenes} scenes. Each scene is one sentence naming the characters, "
        "the conflict and what changes. The first scene must open with a hook, and the last must land a satisfying "
        "twist or payoff that RESOLVES the central conflict.\n"
        'Respond with ONLY JSON: {"scenes": ["...", "..."]}'
    )


def scene_prompt(premise: str, outline: list[str], idx: int, prev: str, feedback: str = "", draft: str = "") -> str:
    is_last = idx == len(outline) - 1
    ending_rule = (
        "This is the FINAL scene: resolve the central conflict and the consequences for every main character. "
        "End on a closing line that lands the emotional payoff. Do NOT end on a cliffhanger or a question."
        if is_last
        else "End on a line that pulls the listener into the next scene."
    )
    revise = ""
    if feedback:
        revise = f"\n\nA reviewer found this problem with your previous draft, fix it:\n{feedback}\n\nPrevious draft:\n{draft}"
    return (
        f"Premise: {premise}\n\nFull outline:\n"
        + "\n".join(f"{j + 1}. {b}" for j, b in enumerate(outline))
        + f"\n\nPrevious scene:\n{prev}\n\nWrite scene {idx + 1} now: {outline[idx]}\n"
        f"Rules: 120-180 words, mostly dialogue, short narrator lines in [brackets]. {ending_rule} "
        f"Output ONLY the scene text.{revise}"
    )


def rubric_prompt(kind: str, text: str, extra: str = "") -> str:
    return (
        f"Rate this {kind} from 1 to 10 on each criterion: hook (does it grab attention), character_consistency "
        "(distinct, consistent voices), pacing, emotional_payoff, cultural_authenticity (feels grounded in Indian "
        "life and idiom, not generic).\n"
        f"{extra}\n---\n{text}\n---\n"
        "Respond with ONLY JSON: {" + ", ".join(f'"{k}": n' for k in RUBRIC)
        + ', "weakest_issue": "one specific sentence on the biggest problem and how to fix it"}'
    )
