from abc import ABC, abstractmethod
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any
import requests

from config.settings import (
    SARVAM_API_KEY,
    SARVAM_STT_ENDPOINT,
    SARVAM_STT_MODEL,
    STT_TIMEOUT_SEC,
    MAX_STT_RETRIES
)
from src.voice.schemas import TranscriptionResult
from src.voice.audio import validate_audio


def normalize_transcript(text: str) -> str:
    """
    Lightweight transcript normalization: strip extra whitespaces, normalize capitalization.
    """
    if not text:
        return ""
    return " ".join(text.strip().split())


class SpeechToText(ABC):
    """
    Abstract Speech-to-Text provider interface.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language_code: str = "auto"
    ) -> TranscriptionResult:
        pass


class SarvamSTT(SpeechToText):
    """
    Sarvam AI Speech-to-Text (STT) implementation.
    Connects to Sarvam's official multilingual speech-to-text endpoint (Saaras).
    Supports English and 10+ Indian languages (Hindi, Tamil, Telugu, Bengali, Assamese, etc.).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = SARVAM_STT_ENDPOINT,
        model: str = SARVAM_STT_MODEL,
        timeout_sec: float = STT_TIMEOUT_SEC,
        max_retries: int = MAX_STT_RETRIES
    ):
        self.api_key = api_key or SARVAM_API_KEY
        self.endpoint = endpoint
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def transcribe(
    self,
    audio_path: str,
    language_code: str = "unknown"
    ) -> TranscriptionResult:
        # Validate audio first
        is_valid, err = validate_audio(audio_path)
        if not is_valid:
            return TranscriptionResult(
                text="",
                error=f"audio_validation_failed: {err}"
            )

        if not self.api_key:
            return TranscriptionResult(
                text="",
                error="SARVAM_API_KEY_NOT_CONFIGURED"
            )

        if language_code == "auto" or not language_code:
            language_code = "unknown"

        headers = {
            "api-subscription-key": self.api_key
        }

        data = {
            "model": self.model,
            "language_code": language_code
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with open(audio_path, "rb") as f:
                    mime_types = {
                        ".wav": "audio/wav",
                        ".webm": "audio/webm",
                        ".ogg": "audio/ogg",
                        ".mp3": "audio/mpeg",
                        ".m4a": "audio/mp4",
                    }

                    mime_type = mime_types.get(
                        Path(audio_path).suffix.lower(),
                        "application/octet-stream"
                    )

                    files = {
                        "file": (Path(audio_path).name, f, mime_type)
                    }

                    response = requests.post(
                        self.endpoint,
                        headers=headers,
                        data=data,
                        files=files,
                        timeout=self.timeout_sec
                    )

                stt_latency_ms = (time.perf_counter() - t0) * 1000

                if response.status_code == 200:
                    resp_json = response.json()
                    transcript = resp_json.get("transcript", "")
                    detected_lang = resp_json.get("language_code", language_code)

                    return TranscriptionResult(
                        text=normalize_transcript(transcript),
                        language=detected_lang,
                        stt_latency_ms=round(stt_latency_ms, 2),
                        confidence=resp_json.get("confidence", 1.0)
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    # If model or language code mismatch, retry with saaras:v2 / unknown
                    if response.status_code in (400, 422) and data.get("language_code") != "unknown":
                        data["language_code"] = "unknown"

            except Exception as e:
                last_error = str(e)

        return TranscriptionResult(
            text="",
            error=f"STT_FAILED: {last_error}"
        )


class MockSarvamSTT(SpeechToText):
    """
    Mock STT provider for deterministic offline testing and benchmark simulation.
    """

    def __init__(
        self,
        simulated_transcript: str = "What causes earthquakes?",
        simulated_lang: str = "en-IN",
        simulated_latency_ms: float = 45.0,
        should_fail: bool = False
    ):
        self.simulated_transcript = simulated_transcript
        self.simulated_lang = simulated_lang
        self.simulated_latency_ms = simulated_latency_ms
        self.should_fail = should_fail

    def transcribe(
        self,
        audio_path: str,
        language_code: str = "auto"
    ) -> TranscriptionResult:
        is_valid, err = validate_audio(audio_path)
        if not is_valid:
            return TranscriptionResult(text="", error=f"audio_validation_failed: {err}")

        if self.should_fail:
            return TranscriptionResult(text="", error="STT_TIMEOUT_OR_NETWORK_ERROR")

        time.sleep(self.simulated_latency_ms / 1000.0)
        return TranscriptionResult(
            text=normalize_transcript(self.simulated_transcript),
            language=self.simulated_lang,
            stt_latency_ms=self.simulated_latency_ms,
            confidence=0.95
        )
