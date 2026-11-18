.PHONY: install test lint demo ablation retrieval

install:
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check src scripts tests

demo:
	python scripts/run_story.py "A greedy merchant is outsmarted by a village girl during a monsoon festival" --judge

ablation:
	python scripts/run_ablation.py

retrieval:
	python scripts/eval_retrieval.py
