#!/usr/bin/env python
"""Generate one story from a premise.

Example:
    python scripts/run_story.py "A greedy merchant is outsmarted by a village girl during a monsoon festival"
    python scripts/run_story.py "..." --no-rag --no-critic --judge
"""
import argparse
import json

from _common import Settings, build_pipeline, setup_logging

from folklore_rag.evals import judge_story


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("premise")
    p.add_argument("--no-rag", action="store_true", help="skip retrieval of reference folktales")
    p.add_argument("--no-critic", action="store_true", help="skip the critic/rewrite loop")
    p.add_argument("--judge", action="store_true", help="score the final story with the judge model")
    p.add_argument("--json", action="store_true", help="print full result as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    setup_logging(a.verbose)
    pipeline, llm, _ = build_pipeline(Settings())
    out = pipeline.run(a.premise, use_rag=not a.no_rag, use_critic=not a.no_critic)
    if a.judge:
        out["judge"] = judge_story(llm, out["story"])

    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        return
    print("RETRIEVED:", [r["title"] for r in out["refs"]])
    print("OUTLINE:", *out["outline"], sep="\n  - ")
    print("SCENE SCORES:", [round(x, 1) for x in out["scene_scores"]])
    print(f"NEAREST CORPUS STORY: {out['nearest_title']} (cosine {out['max_sim']:.3f})\n")
    print(out["story"])
    if a.judge:
        print("\nJUDGE:", json.dumps(out["judge"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
