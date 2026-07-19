"""Thin Gemini wrapper. Fails loudly rather than returning empty answers.

Everything upstream depends only on `complete()`, so swapping providers means
editing this file alone — nothing else in the project knows who serves the model.
"""

import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Pinned deliberately: a "-latest" alias would change model under the evaluation.
# Lite chosen for quota: gemini-3.5-flash allows only 20 requests/DAY free.
MODEL = "gemini-3.1-flash-lite"


MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    pass


def _retry_delay(message: str, default: float = 30.0) -> float:
    """Extract the server's suggested wait from a 429 body, e.g. 'retry in 53.5s'.

    Falls back to `default` when the hint is absent or unparseable — guessing
    short would just burn the next attempt on another 429.
    """
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", message)
    if not match:
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", message)
    return float(match.group(1)) + 2.0 if match else default


def complete(prompt: str, temperature: float = 0.0, max_tokens: int = 700) -> str:
    """Send a prompt, return the completion text. One retry on transient failure.

    Temperature defaults to 0.0 so evaluation runs are reproducible — a
    nondeterministic generator would make the faithfulness scores meaningless.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set — copy .env.example to .env")

    client = genai.Client(api_key=key)
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    # Gemini 3.x charges reasoning tokens against the output
                    # budget, which truncated answers mid-sentence. We want
                    # extraction from supplied context, not reasoning.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text or ""
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise LLMError(
                    f"Gemini call failed after {MAX_ATTEMPTS} attempts: {e}"
                ) from e
            # The free tier allows only 5 requests/minute, so a 429 needs a
            # wait measured in tens of seconds, not the 2s a transient
            # network blip would want. Honour the server's own retry hint.
            wait = _retry_delay(str(e)) if "429" in str(e) else 2.0
            time.sleep(wait)
    raise LLMError("unreachable")
