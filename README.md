# Do LLMs Know When They're Wrong? Metacognition on Medical QA

> This is a learning and portfolio project, **not a clinical tool**. Nothing here should be used for medical decisions.

## Question
When an LLM answers a medical question, does its stated confidence track whether it is correct? This matters for human-AI collaboration: a system that knows when it is uncertain can defer those cases to a clinician.

Two properties are measured separately:
- **Calibration:** is 80% confidence right about 80% of the time? (ECE, Brier score)
- **Metacognitive sensitivity:** is confidence higher on right answers than on wrong ones? (type-2 AUROC)

A model can be well calibrated on average yet unable to tell its right answers from its wrong ones, and vice versa.

## Setup
- **Data:** 200 questions sampled (seed 42) from the test split of [MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (CC-BY-4.0), from Jin et al. (2020), *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams* ([arXiv:2009.13081](https://arxiv.org/abs/2009.13081)).
- **Conditions** (same 200 questions for each model):

| Condition | How it works | Confidence comes from | Calls / question |
| --- | --- | --- | --- |
| A. Verbalized | Answer once and state a 0-100 confidence | The reported number | 1 |
| B. Self-consistency | Answer 5 times; take the majority | Share of samples agreeing with the majority (0.4-1.0) | 5 |
| C. Critic agent | The same model reviews A's answer and reasoning, then gives a final answer and confidence | The critic's revised number | 1 (+ A) |

Prompts are identical across models and ask for JSON only (see [`src/conditions.py`](src/conditions.py)). A response that doesn't parse is retried once; if it fails again it is recorded as a parse error and excluded.

### Methods notes
- **Sampling temperature.** Current Claude models only accept their default sampling settings, so Claude runs at default sampling in every condition rather than temperature 0 / 0.7. For OpenAI, temperature 0 (A, C) and 0.7 (B) are sent when the model accepts them. The settings actually used are saved with every row (`settings` field).
- **Reasoning effort** is set to `low` for both providers (`REASONING_EFFORT` in `.env`).
- **Self-consistency floor.** With 5 votes over 4 options the majority always has at least 2 votes, so B's confidence ranges from 0.4 to 1.0.
- **Significance.** 95% bootstrap intervals (1,000 resamples) are reported for accuracy and AUROC, and comparisons between conditions and models use a paired bootstrap on the same questions ([`results/comparisons.csv`](results/comparisons.csv)).

## Results

## What I learned

## Limitations
- 200 questions: small differences may not be significant
- Exam questions are not real clinical decisions
- Verbalized confidence depends on prompt wording
- Claude could not be run at temperature 0, so its single-answer conditions are not deterministic
- Not a clinical tool; for research and learning only

## Next steps

## Run it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # add your API keys and model names
python src/data.py                # sample 200 questions -> data/sample.jsonl
python src/smoke_test.py          # 3 questions per model, prints raw responses
python src/run.py                 # all models x conditions, cached to results/raw/
python src/analyze.py             # results table + figures
```
`run.py` caches every response, so an interrupted run resumes where it stopped. Use `--providers`, `--conditions` and `--limit` to run a subset, e.g. `python src/run.py --providers claude --conditions A --limit 10`.

## Repo layout
```
src/data.py         load and sample MedQA
src/models.py       one call function per provider (retries on rate limits)
src/conditions.py   prompts, JSON parsing, conditions A/B/C
src/run.py          runs everything, caches JSONL per model x condition
src/metrics.py      accuracy, ECE, Brier, type-2 AUROC, risk-coverage, bootstrap
src/plots.py        the three figures
src/analyze.py      writes results/ tables and figures/
notebooks/analysis.ipynb
```
