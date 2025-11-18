from __future__ import annotations
import json, tempfile
from typing import List, Dict, Any, Tuple
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

# 필요하면 ENV로 바꿔치기
EMBED_MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

class FaissRetriever:
    def __init__(self, index_bytes: bytes, metadata_json: str | bytes, embed_model_name: str = EMBED_MODEL_NAME):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(index_bytes); f.flush()
            self.index = faiss.read_index(f.name)
        if isinstance(metadata_json, (bytes, bytearray)):
            metadata_json = metadata_json.decode("utf-8", errors="ignore")
        meta = json.loads(metadata_json) if isinstance(metadata_json, str) else []
        self.meta: List[Dict[str, Any]] = meta if isinstance(meta, list) else []
        self.embedder = SentenceTransformer(embed_model_name)
        self.metric = self.index.metric_type  # faiss.METRIC_INNER_PRODUCT or L2

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q = self.embedder.encode([query]).astype("float32")
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(q, top_k)
        out = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            rec = self.meta[idx] if 0 <= idx < len(self.meta) else {"id": idx}
            text = rec.get("text") or rec.get("content") or rec.get("body") or ""
            sim = float(score) if self.metric == faiss.METRIC_INNER_PRODUCT else float(1/(1+score))
            out.append({"document": rec.get("document") or rec.get("path") or rec.get("id") or f"idx:{idx}",
                        "text": text, "similarity": sim, "faiss_index": int(idx), "raw_score": float(score)})
        return out
