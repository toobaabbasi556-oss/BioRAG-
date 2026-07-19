"""Thin Gemini wrapper. Fails loudly rather than returning empty answers.

Everything upstream depends only on `complete()`, so swapping providers means
editing this file alone — nothing else in the project knows who serves the model.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Pinned deliberately: a "-latest" alias would change model under the evaluation
# and silently invalidate recorded results. The 2.0 series has no free-tier quota.
MODEL = "gemini-3.5-flash"


class LLMError(RuntimeError):
    pass


def complete(prompt: str, temperature: float = 0.0, max_tokens: int = 700) -> str:
    """Send a prompt, return the completion text. One retry on transient failure.

    Temperature defaults to 0.0 so evaluation runs are reproducible — a
    nondeterministic generator would make the faithfulness scores meaningless.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set — copy .env.example to .env")

    client = genai.Client(api_key=key)
    for attempt in range(2):
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
            if attempt == 1:
                raise LLMError(f"Gemini call failed after retry: {e}") from e
            time.sleep(2)
    raise LLMError("unreachable")
