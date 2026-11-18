#!/usr/bin/env python
"""Export a blind, shuffled CSV (no variant names, no judge scores) for human rating."""
import argparse

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/ablation_results.csv")
    p.add_argument("--out", default="results/to_label.csv")
    a = p.parse_args()
    df = pd.read_csv(a.results)
    df[["id", "premise", "story"]].sample(frac=1, random_state=0).assign(human_score="").to_csv(a.out, index=False)
    print(f"Wrote {a.out}. Fill the human_score column (1-10) and save as results/labeled.csv")


if __name__ == "__main__":
    main()
