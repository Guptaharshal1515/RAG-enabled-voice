import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env manually if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# Configuration settings
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_ENDPOINT = os.getenv(
    "SARVAM_STT_ENDPOINT",
    "https://api.sarvam.ai/speech-to-text"
)
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

# Voice & Latency constraints
MAX_AUDIO_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_BYTES", 25 * 1024 * 1024)) # 25 MB
MAX_AUDIO_DURATION_SEC = float(os.getenv("MAX_AUDIO_DURATION_SEC", 60.0))       # 60s
STT_TIMEOUT_SEC = float(os.getenv("STT_TIMEOUT_SEC", 5.0))
MAX_STT_RETRIES = int(os.getenv("MAX_STT_RETRIES", 1))
