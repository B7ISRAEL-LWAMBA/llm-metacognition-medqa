"""One call function per provider. Each takes a prompt and returns text.

Retries: both SDKs retry rate-limit (429), overload/5xx and connection
errors with exponential backoff. We raise max_retries from the default 2 to 6
so a long run survives bursts of rate limiting.

Sampling: current Claude models (e.g. Haiku 5.5) reject non-default
temperature, so Claude always runs at its default sampling. For OpenAI we send
the requested temperature, and if the model rejects it (reasoning models do)
we drop it for the rest of the run. Every call returns the settings actually
used so they are saved with each result.
"""

import os

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
MAX_TOKENS = 4000  # room for thinking plus a short JSON answer
MAX_RETRIES = 6

_anthropic_client = None
_openai_client = None
# Parameters a model has rejected once; we stop sending them.
_openai_unsupported: set[str] = set()


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    return _anthropic_client


def _openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(max_retries=MAX_RETRIES)
    return _openai_client


def call_claude(prompt: str, temperature: float | None = None, model: str = CLAUDE_MODEL) -> tuple[str, dict]:
    """Send one prompt to Claude. `temperature` is accepted for a uniform
    interface but not sent, since current Claude models reject it."""
    response = _anthropic().messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": REASONING_EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    settings = {"model": response.model, "temperature": "default", "effort": REASONING_EFFORT,
                "stop_reason": response.stop_reason}
    if response.stop_reason == "refusal":
        return "", settings
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, settings


def call_openai(prompt: str, temperature: float | None = None, model: str = OPENAI_MODEL) -> tuple[str, dict]:
    """Send one prompt to OpenAI via the Responses API."""
    params = {"model": model, "input": prompt, "max_output_tokens": MAX_TOKENS}
    if temperature is not None and "temperature" not in _openai_unsupported:
        params["temperature"] = temperature
    if "reasoning" not in _openai_unsupported:
        params["reasoning"] = {"effort": REASONING_EFFORT}

    while True:
        try:
            response = _openai().responses.create(**params)
            break
        except openai.BadRequestError as e:
            # Drop a parameter this model doesn't support and try again.
            rejected = next((p for p in ("temperature", "reasoning") if p in params and p in str(e)), None)
            if rejected is None:
                raise
            print(f"[models] {model} rejected '{rejected}'; continuing without it.")
            _openai_unsupported.add(rejected)
            params.pop(rejected)

    settings = {"model": response.model, "temperature": params.get("temperature", "default"),
                "effort": params.get("reasoning", {}).get("effort", "n/a"), "stop_reason": response.status}
    return response.output_text, settings


PROVIDERS = {"claude": (call_claude, CLAUDE_MODEL), "openai": (call_openai, OPENAI_MODEL)}


def call_model(provider: str, prompt: str, temperature: float | None = None) -> tuple[str, dict]:
    fn, _ = PROVIDERS[provider]
    return fn(prompt, temperature=temperature)
