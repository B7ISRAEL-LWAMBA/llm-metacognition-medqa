"""Load MedQA (USMLE, 4 options) and draw a fixed random sample.

Dataset: GBaker/MedQA-USMLE-4-options on Hugging Face (CC-BY-4.0),
derived from Jin et al. 2020, "What Disease does this Patient Have?" (arXiv:2009.13081).
Fields used: question, options {A, B, C, D}, answer_idx.
"""

import json
import random
from pathlib import Path

DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.jsonl"


def build_sample(n: int = 200, seed: int = 42, path: Path = SAMPLE_PATH) -> list[dict]:
    """Sample n test questions with a fixed seed and save them to JSONL."""
    from datasets import load_dataset  # imported here so other modules load fast

    test = load_dataset(DATASET_NAME, split="test")
    # Sample row indices rather than shuffling the dataset, so the chosen
    # question_ids are easy to inspect and reproduce.
    indices = sorted(random.Random(seed).sample(range(len(test)), n))

    rows = []
    for i in indices:
        q = test[i]
        rows.append({
            "question_id": i,  # row index in the test split
            "question": q["question"],
            "options": q["options"],
            "correct_answer": q["answer_idx"],
            "meta_info": q["meta_info"],  # step1 or step2&3
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    """Load the saved sample, building it first if it doesn't exist."""
    if not path.exists():
        return build_sample(path=path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    rows = build_sample()
    print(f"Saved {len(rows)} questions to {SAMPLE_PATH}")
    counts = {}
    for r in rows:
        counts[r["correct_answer"]] = counts.get(r["correct_answer"], 0) + 1
    print("Correct-answer distribution:", dict(sorted(counts.items())))
