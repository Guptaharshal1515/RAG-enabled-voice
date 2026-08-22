import json
import pandas as pd

df = pd.read_parquet("data/processed/chunks.parquet")
unique_queries = df[["query_id", "language", "text"]].drop_duplicates(subset=["query_id"]).head(100)

queries = []
# 1. Dataset queries
for idx, row in unique_queries.iterrows():
    text_sample = str(row["text"])[:60].replace("\n", " ")
    queries.append({
        "id": f"q_data_{row['query_id']}",
        "query": text_sample,
        "language": str(row["language"])
    })

# 2. Domain & Multilingual queries
extra_queries = [
    {"id": "q_eng_01", "query": "What is a corporation in law?", "language": "en"},
    {"id": "q_eng_02", "query": "What causes earthquakes?", "language": "en"},
    {"id": "q_eng_03", "query": "How does photosynthesis work in plants?", "language": "en"},
    {"id": "q_eng_04", "query": "What is CRISPR Cas9 gene editing?", "language": "en"},
    {"id": "q_eng_05", "query": "What are tectonic plates?", "language": "en"},
    {"id": "q_eng_06", "query": "How do vaccines protect human health?", "language": "en"},
    {"id": "q_eng_07", "query": "What is machine learning in artificial intelligence?", "language": "en"},
    {"id": "q_eng_08", "query": "How does gravity work according to Einstein?", "language": "en"},
    {"id": "q_as_01", "query": "কৰ্পোৰেচন কি?", "language": "as"},
    {"id": "q_as_02", "query": "ব্যৱসায় কেনেকৈ আৰম্ভ কৰিব পাৰি?", "language": "as"},
    {"id": "q_as_03", "query": "নিগমৰ সংজ্ঞা কি?", "language": "as"},
    {"id": "q_hi_01", "query": "भूकंप कैसे आते हैं?", "language": "hi"},
    {"id": "q_hi_02", "query": "प्रकाश संश्लेषण क्या है?", "language": "hi"},
    {"id": "q_ta_01", "query": "நிலநடுக்கம் எவ்வாறு ஏற்படுகிறது?", "language": "ta"},
    {"id": "q_te_01", "query": "భూకంపాలు ఎలా సంభవిస్తాయి?", "language": "te"}
]
queries.extend(extra_queries)

with open("evaluation/queries.jsonl", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Successfully generated {len(queries)} benchmark queries in evaluation/queries.jsonl")
