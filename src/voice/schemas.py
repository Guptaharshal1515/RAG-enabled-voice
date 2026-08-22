from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from src.generation.schemas import RAGResponse


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = "en-IN"
    stt_latency_ms: float = 0.0
    duration_ms: Optional[float] = None
    confidence: Optional[float] = 1.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceRAGResponse:
    request_id: str
    transcription: TranscriptionResult
    rag_response: RAGResponse
    latencies_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "transcription": self.transcription.to_dict(),
            "rag_response": self.rag_response.to_dict(),
            "latencies_ms": self.latencies_ms
        }
