"""Day-1 check: call each model on 3 questions and print the raw responses.

Usage: python src/smoke_test.py [claude|openai]
Costs a few cents. Nothing is saved.
"""

import sys

from conditions import format_answer_prompt, parse_json
from data import load_sample
from models import PROVIDERS, call_model

providers = sys.argv[1:] or list(PROVIDERS)
questions = load_sample()[:3]

for provider in providers:
    print(f"\n===== {provider} ({PROVIDERS[provider][1]}) =====")
    for q in questions:
        raw, settings = call_model(provider, format_answer_prompt(q), temperature=0)
        parsed = parse_json(raw, "answer")
        print(f"\n--- question {q['question_id']} (correct: {q['correct_answer']}) ---")
        print("settings:", settings)
        print("raw response:", raw)
        print("parsed:", parsed)
