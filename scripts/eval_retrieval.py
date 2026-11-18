#!/usr/bin/env python
"""Compare embedding models on recall@k. Needs no API key."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from folklore_rag.config import EMBEDDING_MODELS, Settings
from folklore_rag.corpus import load_corpus
from folklore_rag.evals import recall_at_k
from folklore_rag.retrieval import Retriever


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-k", type=int, default=3)
    a = p.parse_args()
    retriever = Retriever(load_corpus(Settings().corpus_path))
    for key, name in EMBEDDING_MODELS.items():
        print(f"{name}: recall@{a.k} = {recall_at_k(retriever, key, a.k):.2f}")


if __name__ == "__main__":
    main()
