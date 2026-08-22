from typing import Dict, Any, List, Optional
import os
import pandas as pd


class MetadataStore:
    """
    Metadata storage decoupled from FAISS vectors.
    Maps integer vector IDs (0, 1, 2...) to complete chunk dictionaries/records.
    """

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self.records: List[Dict[str, Any]] = records or []

    def add(self, record: Dict[str, Any]) -> int:
        vector_id = len(self.records)
        record["vector_id"] = vector_id
        self.records.append(record)
        return vector_id

    def get(self, vector_id: int) -> Optional[Dict[str, Any]]:
        if 0 <= vector_id < len(self.records):
            return self.records[vector_id]
        return None

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str) -> "MetadataStore":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found at '{path}'")
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        return cls(records)
