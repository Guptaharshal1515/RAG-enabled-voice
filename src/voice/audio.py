import wave
import struct
import math
from pathlib import Path
from typing import Tuple, Optional
from config.settings import MAX_AUDIO_SIZE_BYTES, MAX_AUDIO_DURATION_SEC

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}


def validate_audio(
    path: str,
    max_size_bytes: int = MAX_AUDIO_SIZE_BYTES,
    max_duration_sec: float = MAX_AUDIO_DURATION_SEC
) -> Tuple[bool, Optional[str]]:
    """
    Validates audio file existence, format extension, non-empty size, and duration constraints.
    Returns (is_valid, error_reason).
    """
    if not path:
        return False, "empty_path"

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return False, "file_not_found"

    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"unsupported_format_{ext}"

    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "empty_audio_file"

    if file_size > max_size_bytes:
        return False, "audio_file_too_large"

    # For WAV files, inspect duration header
    if ext == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                if duration > max_duration_sec:
                    return False, "audio_duration_exceeded"
        except Exception:
            pass # Non-standard WAV or partial header, allow downstream STT inspection

    return True, None


def generate_synthetic_wav(
    output_path: str,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0
) -> str:
    """
    Utility to create a valid synthetic WAV file for unit tests and local benchmarking.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1) # Mono
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        
        raw_frames = bytearray()
        for i in range(num_samples):
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            raw_frames.extend(struct.pack("<h", val))
        wf.writeframes(raw_frames)

    return output_path
