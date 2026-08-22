from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    query_id: Optional[int]
    split: str
    language: str
    text: str
    chunk_index: int
    chunk_strategy: str
    token_count: int
    parent_document: Optional[str] = None
    is_gold_passage: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
