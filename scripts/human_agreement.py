#!/usr/bin/env python
"""Spearman correlation between human ratings and the LLM judge, plus failure analysis."""
import argparse

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/ablation_results.csv")
    p.add_argument("--labels", default="results/labeled.csv")
    a = p.parse_args()

    df = pd.read_csv(a.results)
    lab = pd.read_csv(a.labels).dropna(subset=["human_score"])
    merged = lab.merge(df[["id", "variant", "judge_overall"]], on="id").dropna(subset=["judge_overall"])
    if len(merged) < 3:
        raise SystemExit(f"Need at least 3 labeled rows, found {len(merged)}. Did you fill human_score?")

    rho, pval = spearmanr(merged["human_score"], merged["judge_overall"])
    print(f"Spearman rho = {rho:.2f} (p = {pval:.3f}, n = {len(merged)})\n")
    print("Mean human score per variant:")
    print(merged.groupby("variant")["human_score"].mean().round(2).to_string())

    merged["gap"] = merged["judge_overall"] - merged["human_score"]
    print("\nJudge liked it, you did not:")
    print(merged.sort_values("gap", ascending=False).head(3)[["id", "judge_overall", "human_score"]].to_string(index=False))
    print("\nHighest corpus similarity (possible copying):")
    print(df.sort_values("max_sim", ascending=False).head(3)[["id", "max_sim", "nearest"]].to_string(index=False))


if __name__ == "__main__":
    main()
