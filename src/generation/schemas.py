from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Source:
    chunk_id: str
    document_id: str
    score: float
    text: Optional[str] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredAnswer:
    answer: str
    source_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class RAGResponse:
    request_id: str
    query: str
    answer: str
    sources: List[Source]
    retrieval_method: str = "hybrid_rrf"
    grounded: bool = False
    refusal: bool = False
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() if isinstance(s, Source) else s for s in self.sources],
            "retrieval_method": self.retrieval_method,
            "grounded": self.grounded,
            "refusal": self.refusal,
            "latencies_ms": self.latencies_ms,
            "error": self.error
        }
