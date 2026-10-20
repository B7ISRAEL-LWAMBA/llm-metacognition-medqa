"""Run every model x condition over the sample, caching each row to JSONL.

Usage:
    python src/run.py                          # everything
    python src/run.py --providers claude --conditions A --limit 10

Each (model, condition) pair writes results/raw/<model>__<condition>.jsonl.
Rows are appended as soon as they finish, and questions already in the file
are skipped, so a crashed run picks up where it left off without paying twice.
Condition C reads condition A's file, so A always runs before C.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from conditions import run_critic, run_self_consistency, run_verbalized
from data import load_sample
from models import PROVIDERS

RAW_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"


def raw_path(provider: str, condition: str) -> Path:
    _, model = PROVIDERS[provider]
    return RAW_DIR / f"{model}__{condition}.jsonl"


def read_rows(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["question_id"]: r for r in rows}


def run_condition(provider: str, condition: str, questions: list[dict], workers: int) -> None:
    path = raw_path(provider, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows(path)
    todo = [q for q in questions if q["question_id"] not in done]
    label = f"{path.stem}"
    if not todo:
        print(f"{label}: all {len(questions)} cached")
        return

    if condition == "C":
        a_rows = read_rows(raw_path(provider, "A"))
        todo = [q for q in todo if q["question_id"] in a_rows]
        task = lambda q: run_critic(q, provider, a_rows[q["question_id"]])
    elif condition == "B":
        task = lambda q: run_self_consistency(q, provider)
    else:
        task = lambda q: run_verbalized(q, provider)

    lock = threading.Lock()
    errors = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(task, q): q for q in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
            try:
                row = fut.result()
            except Exception as e:  # leave the question uncached; a rerun retries it
                errors += 1
                tqdm.write(f"  q{futures[fut]['question_id']} failed: {type(e).__name__}: {e}")
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
    if errors:
        print(f"{label}: {errors} questions failed with API errors. Rerun to retry them.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=list(PROVIDERS), choices=list(PROVIDERS))
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--workers", type=int, default=4, help="parallel API calls")
    args = parser.parse_args()

    questions = load_sample()[: args.limit]
    for provider in args.providers:
        for condition in sorted(args.conditions):  # A before C
            run_condition(provider, condition, questions, args.workers)


if __name__ == "__main__":
    main()
