"""Thin Groq wrapper. Fails loudly rather than returning empty answers."""

import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


class LLMError(RuntimeError):
    pass


def complete(prompt: str, temperature: float = 0.0, max_tokens: int = 700) -> str:
    """Send a prompt, return the completion text. One retry on transient failure."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set — copy .env.example to .env")

    client = Groq(api_key=key)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == 1:
                raise LLMError(f"Groq call failed after retry: {e}") from e
            time.sleep(2)
    raise LLMError("unreachable")
