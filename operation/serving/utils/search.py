from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import json
import os


embedder = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

INDEX_DIR = "C:/Users/dm_ohminchan/Model/operation/Vector/index"
VECTOR_INDEX_PATH = os.path.join(INDEX_DIR, "vector.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

faiss_index = faiss.read_index(VECTOR_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def vector_search(query: str, top_k: int = 15) -> List[Dict]:
    query_vec = embedder.encode([query])
    query_vec = normalize(query_vec, axis=1).astype("float32")
    D, I = faiss_index.search(query_vec, top_k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(metadata):
            results.append({
                **metadata[i],
                "text": metadata[i].get("text", ""),
                "similarity": float(1 - score)  # L2 → Cosine 유사도로 변환
            })
    return results
