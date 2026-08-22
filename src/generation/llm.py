from abc import ABC, abstractmethod
import time
import json
import re
from typing import Optional, Dict, Any


class LLM(ABC):
    """
    Abstract LLM Interface.
    Enables swappable LLM providers (Local Fast Engine, Sarvam, Gemini, OpenAI, etc.).
    """

    @abstractmethod
    def generate(self, prompt: str, timeout_sec: float = 5.0) -> str:
        """
        Generate text response for a given prompt.
        """
        pass


def generate_with_retry(
    llm: LLM,
    prompt: str,
    max_retries: int = 2,
    timeout_sec: float = 5.0,
    backoff_factor: float = 0.3
) -> str:
    """
    Execute LLM generation with automatic retry and timeout handling.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = llm.generate(prompt, timeout_sec=timeout_sec)
            if response and response.strip():
                return response
            raise ValueError("LLM returned an empty response string.")
        except Exception as err:
            last_error = err
            if attempt < max_retries:
                sleep_duration = backoff_factor * (2 ** attempt)
                time.sleep(sleep_duration)

    raise RuntimeError(f"LLM generation failed after {max_retries + 1} attempts: {last_error}")
