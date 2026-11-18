"""Shared bootstrap for CLI scripts."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from folklore_rag.config import Settings
from folklore_rag.corpus import load_corpus
from folklore_rag.llm import LLMClient
from folklore_rag.pipeline import StoryPipeline
from folklore_rag.retrieval import Retriever


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_pipeline(settings: Settings):
    corpus = load_corpus(settings.corpus_path)
    retriever = Retriever(corpus, keys=[settings.embedding])
    llm = LLMClient(settings)
    return StoryPipeline(settings, llm, retriever), llm, retriever
