# FolkloreRAG



A **LangGraph + Qdrant** pipeline that turns a one-line premise into an original 4-scene Indian audio-drama script, with a critic/rewrite loop and **LLM-as-judge evals**. Built on Groq's free API and runnable on a laptop or free Colab.

## How it works

```
retrieve -> plan -> write_scene -> critic --(score < 7 and rewrites left)--> write_scene
                                     |
                                     +--> advance --(more scenes)--> write_scene
                                                  +--(done)--> originality_check -> END
```

1. **Retrieve**: top-3 similar folktales from Qdrant (multilingual embeddings, English and Hindi queries), used for structure and tone only.
2. **Plan**: writer model outlines 4 scenes.
3. **Write + critique**: each scene is scored on hook, character consistency, pacing, emotional payoff and cultural authenticity; below 7.0 it is rewritten with the critic's feedback (max 2 rewrites). The final scene has its own prompt that requires resolution instead of a cliffhanger.
4. **Originality check**: cosine similarity between the finished story and its nearest corpus story, to flag copying.
5. **Judge**: a separate model family (when available) scores the finished story on the same rubric.

## Repo layout

```
src/folklore_rag/   config, llm (cache + retries), retrieval, prompts, pipeline, evals
scripts/            run_story, run_ablation, eval_retrieval, export_for_labeling, human_agreement
tests/              unit tests + mocked end-to-end graph test (no API key needed)
data/               corpus.json (12 public-domain tale summaries), premises.txt
notebooks/          original exploratory Colab notebook
```

## Quickstart

```bash
git clone https://github.com/Harsh3231qubit/folklore-rag && cd folklore-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your free GROQ_API_KEY

make test                                            # no API key needed
python scripts/eval_retrieval.py                     # recall@3 per embedding model
python scripts/run_story.py "A greedy merchant is outsmarted by a village girl during a monsoon festival" --judge
python scripts/run_ablation.py                       # 4 variants x premises -> results/ablation_results.csv
```

Every LLM call is cached under `.cache/llm`, so interrupted or rate-limited runs resume where they stopped. Models are auto-discovered from your key; override with `WRITER_MODEL`, `CRITIC_MODEL`, `JUDGE_MODEL` in `.env`.

## Evaluation workflow

| Step | Command |
|---|---|
| Retrieval recall@k, two embedding models | `scripts/eval_retrieval.py` |
| 2x2 ablation (RAG x critic) judged on the rubric | `scripts/run_ablation.py` |
| Blind sheet for human rating | `scripts/export_for_labeling.py` |
| Judge vs human Spearman + failure analysis | `scripts/human_agreement.py` |

## Results so far

An early notebook run (5 premises x 4 variants = 20 stories, before the final-scene fix) gave judge averages of 3.5 to 4.6 out of 10:

| Variant | judge_overall |
|---|---|
| critic_only | 4.64 |
| no_rag_no_critic | 4.16 |
| rag_only | 4.00 |
| rag_plus_critic | 3.52 |

With 5 premises these gaps are within noise, so I do not claim any variant wins. The judge's main complaints were weak pacing and unresolved endings, which is what the final-scene prompt now targets; re-run `run_ablation.py` to measure it. Human-vs-judge agreement has not been measured yet.

## Limitations

- Corpus is 12 paraphrased summaries; recall@3 of 1.0 is therefore not informative.
- Small sample (5 premises) and an LLM judge that shares a rubric with the critic.
- Output is text only; no audio or TTS stage.


