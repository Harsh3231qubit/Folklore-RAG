#!/usr/bin/env python
"""2x2 ablation (RAG x critic) over a list of premises, scored by the LLM judge.

Results are cached per LLM call, so re-running resumes where a rate limit stopped it.

Example:
    python scripts/run_ablation.py --premises data/premises.txt --out results/ablation_results.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from _common import Settings, build_pipeline, setup_logging

from folklore_rag.config import RUBRIC
from folklore_rag.evals import judge_story

VARIANTS = {
    "no_rag_no_critic": (False, False),
    "rag_only": (True, False),
    "critic_only": (False, True),
    "rag_plus_critic": (True, True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--premises", default="data/premises.txt", help="text file, one premise per line")
    p.add_argument("--out", default="results/ablation_results.csv")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    setup_logging(a.verbose)
    premises = [l.strip() for l in Path(a.premises).read_text(encoding="utf-8").splitlines() if l.strip()]
    pipeline, llm, _ = build_pipeline(Settings())

    rows = []
    for pid, premise in enumerate(premises):
        for name, (rag, critic) in VARIANTS.items():
            out = pipeline.run(premise, use_rag=rag, use_critic=critic)
            ev = judge_story(llm, out["story"])
            rows.append({"id": f"{pid}_{name}", "premise": premise, "variant": name, "story": out["story"],
                         "max_sim": out["max_sim"], "nearest": out["nearest_title"], **ev})
            print(f"done {pid} {name} judge={ev['judge_overall']}")

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summary = df.groupby("variant")[RUBRIC + ["judge_overall", "max_sim"]].mean().round(2).sort_values("judge_overall", ascending=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
