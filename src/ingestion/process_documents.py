import json
import os
import sys
from typing import Optional
import pandas as pd
from datasets import load_dataset

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.chunking.adaptive_chunker import AdaptiveChunker
from src.chunking.token_counter import estimate_tokens
from src.chunking.analyze_chunks import analyze_chunks

DATASET_NAME = "ai4bharat/MSMARCO-XI"


def process_dataset(
    max_records: Optional[int] = None,
    split: str = "validation",
    streaming: bool = True
):
    """
    Process MSMARCO-XI records using Adaptive Multi-Strategy Chunking.
    Extracts Indic and English passages, maps metadata, applies adaptive chunking,
    and saves the output to data/processed/chunks.parquet and chunks.json.
    """
    print(f"Loading '{DATASET_NAME}' (split={split}, streaming={streaming})...")
    dataset = load_dataset(DATASET_NAME, split=split, streaming=streaming)

    chunker = AdaptiveChunker(
        short_threshold=200,
        medium_threshold=800,
        long_threshold=2000
    )

    processed_chunks = []
    processed_records = 0

    for index, record in enumerate(dataset):
        if max_records and processed_records >= max_records:
            break

        query_id = record.get("query_id", index)
        target_lang = record.get("target_lang", "indic")
        passages_obj = record.get("passages", {})

        trans_passages = passages_obj.get("Translated_passages", [])
        is_selected = passages_obj.get("is_selected", [])

        # Process each translated passage
        for p_idx, text in enumerate(trans_passages):
            if not text or not text.strip():
                continue

            doc_id = f"doc_{query_id}_p{p_idx}"
            is_gold = bool(is_selected[p_idx]) if p_idx < len(is_selected) else False

            result = chunker.chunk(text)

            for chunk_idx, chunk_text in enumerate(result["chunks"]):
                chunk_id = f"{doc_id}_c{chunk_idx}"
                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "query_id": query_id,
                    "split": split,
                    "language": target_lang,
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": result["strategy"],
                    "token_count": estimate_tokens(chunk_text),
                    "is_gold_passage": is_gold,
                    "parent_document": doc_id
                })

        processed_records += 1
        if processed_records % 100 == 0:
            print(f"Processed {processed_records} query records ({len(processed_chunks)} chunks created)...")

    os.makedirs("data/processed", exist_ok=True)

    # Save to Parquet
    parquet_path = "data/processed/chunks.parquet"
    df = pd.DataFrame(processed_chunks)
    df.to_parquet(parquet_path, index=False)

    # Save to JSON
    json_path = "data/processed/chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete:")
    print(f"  Total records processed : {processed_records}")
    print(f"  Total chunks generated  : {len(processed_chunks)}")
    print(f"  Saved Parquet to        : {parquet_path}")
    print(f"  Saved JSON to           : {json_path}")

    # Run analytics
    analyze_chunks(processed_chunks)

    return processed_chunks


if __name__ == "__main__":
    # Process initial test subset of 100 validation queries to verify pipeline cleanly
    process_dataset(max_records=100, split="validation", streaming=True)
