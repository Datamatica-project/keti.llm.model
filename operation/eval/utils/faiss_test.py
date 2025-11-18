# operation/eval/utils/faiss_test.py
from __future__ import annotations
from typing import List, Dict
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

BASE_DIR = r"C:\Users\dm_ohminchan\Model\operation\Vector\index"
INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
META_PATH  = os.path.join(BASE_DIR, "metadata.json")

EMBED_MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

_embedder = None
_faiss_index = None
_metadata = None
_metric_type = None  # faiss.METRIC_INNER_PRODUCT or L2


def _ensure_loaded():
    """인덱스, 메타, 임베더를 한 번만 로드"""
    global _embedder, _faiss_index, _metadata, _metric_type

    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)

    if _faiss_index is None or _metadata is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f" 인덱스 파일이 없습니다: {INDEX_PATH}")
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f" 메타데이터 파일이 없습니다: {META_PATH}")

        print(f" 로컬 인덱스 로드 중: {INDEX_PATH}")
        _faiss_index = faiss.read_index(INDEX_PATH)
        _metric_type = _faiss_index.metric_type

        with open(META_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        print(f" 인덱스 로드 완료 (entries={len(_metadata)})")


def _encode_query(query: str) -> np.ndarray:
    """SentenceTransformer 임베딩"""
    vec = _embedder.encode([query])
    return vec.astype("float32")



def vector_search(query: str, top_k: int = 10) -> List[Dict]:
    """로컬 인덱스 기반 벡터 검색"""
    _ensure_loaded()

    q = _encode_query(query)

    # 인덱스가 Inner Product(IP)인 경우 → 코사인 유사도처럼 정규화
    if _metric_type == faiss.METRIC_INNER_PRODUCT:
        q = normalize(q, axis=1).astype("float32")

    D, I = _faiss_index.search(q, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        meta = _metadata[idx] if 0 <= idx < len(_metadata) else {}
        text = meta.get("text") or meta.get("content") or meta.get("body") or ""
        # 점수 변환
        if _metric_type == faiss.METRIC_INNER_PRODUCT:
            similarity = float(score)  # [-1, 1] 근처
        else:
            similarity = float(1 / (1 + score))  # L2 distance → 유사도로 변환
        results.append({
            "document": meta.get("document") or meta.get("path") or meta.get("id") or f"idx:{idx}",
            "text": text,
            "similarity": similarity,
            "raw_score": float(score),
            "faiss_index": int(idx),
        })

    return results