"""The three experimental conditions.

A. Verbalized confidence: one answer plus a self-reported 0-100 confidence.
B. Self-consistency: 5 samples; confidence = share agreeing with the majority.
C. Critic agent: the same model reviews A's answer and gives a final answer
   and revised confidence.

Each function returns one result row (a dict) for one question.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone

from models import call_model

N_SAMPLES = 5
SC_TEMPERATURE = 0.7  # only applied where the provider accepts it
VALID = {"A", "B", "C", "D"}

ANSWER_PROMPT = """Answer the following medical multiple-choice question.

Question: {question}
Options:
A. {A}
B. {B}
C. {C}
D. {D}

Think briefly, then respond with ONLY a JSON object, no other text:
{{"reasoning": "<2-3 sentences>", "answer": "<A, B, C, or D>", "confidence": <integer 0-100, your probability that your answer is correct>}}"""

CRITIC_PROMPT = """You are reviewing another model's answer to a medical multiple-choice question.

Question: {question}
Options: {options}

Proposed answer: {answer}
Reasoning: {reasoning}
Stated confidence: {confidence}

Check the reasoning for errors. Keep the answer if it is correct; change it if it is not. Then give your own confidence.
Respond with ONLY a JSON object:
{{"critique": "<2-3 sentences>", "final_answer": "<A, B, C, or D>", "confidence": <integer 0-100>}}"""


def format_answer_prompt(q: dict) -> str:
    return ANSWER_PROMPT.format(question=q["question"], **q["options"])


def format_options(options: dict) -> str:
    return " ".join(f"{k}. {v}" for k, v in options.items())


def parse_json(text: str, answer_key: str) -> dict | None:
    """Pull the JSON object out of a response; return None if it's unusable.

    Strips code fences, then falls back to the outermost {...} in case the
    model added text around it. Validates the answer letter and confidence.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    answer = str(data.get(answer_key, "")).strip().upper()[:1]
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if answer not in VALID or not 0 <= confidence <= 100:
        return None
    data[answer_key] = answer
    data["confidence"] = confidence
    return data


def ask(provider: str, prompt: str, answer_key: str, temperature: float | None) -> tuple[dict | None, str, dict]:
    """Call the model; if the JSON doesn't parse, retry once."""
    for _ in range(2):
        raw, settings = call_model(provider, prompt, temperature=temperature)
        parsed = parse_json(raw, answer_key)
        if parsed is not None:
            return parsed, raw, settings
    return None, raw, settings


def _row(q: dict, provider: str, condition: str, settings: dict, raw, **fields) -> dict:
    answer = fields.pop("answer", None)
    row = {
        "question_id": q["question_id"],
        "model": settings.get("model"),
        "provider": provider,
        "condition": condition,
        "answer": answer,
        "correct_answer": q["correct_answer"],
        "is_correct": None if answer is None else answer == q["correct_answer"],
        "parse_error": answer is None,
        "raw_response": raw,
        "settings": settings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row.update(fields)
    return row


def run_verbalized(q: dict, provider: str) -> dict:
    """Condition A."""
    parsed, raw, settings = ask(provider, format_answer_prompt(q), "answer", temperature=0)
    if parsed is None:
        return _row(q, provider, "A", settings, raw, answer=None, confidence=None)
    return _row(q, provider, "A", settings, raw, answer=parsed["answer"],
                confidence=parsed["confidence"] / 100, reasoning=parsed.get("reasoning"))


def run_self_consistency(q: dict, provider: str) -> dict:
    """Condition B. Confidence = majority votes / 5 (from 0.2 to 1.0).

    Samples that fail to parse count as non-votes, so they lower confidence
    rather than being silently dropped. Ties go to the answer seen first.
    """
    prompt = format_answer_prompt(q)
    samples, raws, settings = [], [], {}
    for _ in range(N_SAMPLES):
        parsed, raw, settings = ask(provider, prompt, "answer", temperature=SC_TEMPERATURE)
        samples.append(parsed)
        raws.append(raw)

    votes = [s["answer"] for s in samples if s is not None]
    if not votes:
        return _row(q, provider, "B", settings, raws, answer=None, confidence=None)
    majority, count = Counter(votes).most_common(1)[0]
    verbal = [s["confidence"] / 100 for s in samples if s is not None]
    return _row(q, provider, "B", settings, raws, answer=majority,
                confidence=count / N_SAMPLES, votes=votes,
                n_valid_samples=len(votes),
                mean_verbalized_confidence=sum(verbal) / len(verbal))


def run_critic(q: dict, provider: str, a_row: dict) -> dict:
    """Condition C. Reuses condition A's answer for the same question and model."""
    if a_row["parse_error"]:
        return _row(q, provider, "C", a_row["settings"], None, answer=None, confidence=None,
                    initial_answer=None, changed_answer=None)
    prompt = CRITIC_PROMPT.format(
        question=q["question"], options=format_options(q["options"]),
        answer=a_row["answer"], reasoning=a_row.get("reasoning", ""),
        confidence=round(a_row["confidence"] * 100),
    )
    parsed, raw, settings = ask(provider, prompt, "final_answer", temperature=0)
    if parsed is None:
        return _row(q, provider, "C", settings, raw, answer=None, confidence=None,
                    initial_answer=a_row["answer"], changed_answer=None)
    final = parsed["final_answer"]
    return _row(q, provider, "C", settings, raw, answer=final,
                confidence=parsed["confidence"] / 100,
                critique=parsed.get("critique"),
                initial_answer=a_row["answer"],
                initial_confidence=a_row["confidence"],
                changed_answer=final != a_row["answer"])
