from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .search import vector_search        # RAG 검색 함수
from .reranker import load_reranker, rerank_with_bge   # BGE reranker


# ===== 설정 =====
# 입력/출력 경로
INPUT_QA_PATH = "C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/data/instrcution/qa_with_perspectives_cleaned.json"
OUTPUT_QA_PATH = "C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/data/instrcution/qa_with_perspectives_cleaned_gt.json"

# Retrieval 설정
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 8

# Answer-Chunk 유사도 Threshold
ANSWER_CHUNK_SIM_THRESHOLD = 0.45

# 임베딩 모델 (검색기와 동일)
EMBED_MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"


# ===== 유틸 함수 =====

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def load_qa_dataset(path: str) -> List[Dict[str, Any]]:
    """QA JSON 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 내부가 2중 리스트라면 평탄화
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for sub in data:
            flat.extend(sub)
        return flat

    return data


def build_ground_truth_for_item(
    item: Dict[str, Any],
    embedder: SentenceTransformer,
    reranker_pipeline
) -> Dict[str, Any]:
    """QA item 하나에 대해 Ground Truth chunk 생성"""

    question = item.get("question", item.get("QUESTION", "")).strip()
    answer = item.get("answer", item.get("ANSWER", "")).strip()

    if not question or not answer:
        item["ground_truth_chunk_ids"] = []
        item["ground_truth_chunks"] = []
        return item

    # 1) 검색 수행
    try:
        vector_results = vector_search(question, top_k=RETRIEVAL_TOP_K)
    except Exception as e:
        print(f"[WARN] vector_search 실패: {e}")
        item["ground_truth_chunk_ids"] = []
        item["ground_truth_chunks"] = []
        return item

    if not vector_results:
        item["ground_truth_chunk_ids"] = []
        item["ground_truth_chunks"] = []
        return item

    # 2) rerank 수행
    try:
        reranked = rerank_with_bge(
            query=question,
            docs=vector_results,
            reranker_pipeline=reranker_pipeline,
            top_k=RERANK_TOP_K
        )
    except Exception as e:
        print(f"[WARN] reranker 실패: {e}")
        item["ground_truth_chunk_ids"] = []
        item["ground_truth_chunks"] = []
        return item

    # 3) answer와 chunk 유사도 계산
    answer_vec = embedder.encode([answer])[0]

    gt_ids = []
    gt_chunks = []

    for doc, rerank_score in reranked:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        chunk_vec = embedder.encode([text])[0]
        sim = cosine_similarity(answer_vec, chunk_vec)

        if sim >= ANSWER_CHUNK_SIM_THRESHOLD:
            chunk_id = doc.get("chunk_id") or f"{doc.get('document')}::{doc.get('index')}"

            gt_ids.append(chunk_id)
            gt_chunks.append({
                "chunk_id": chunk_id,
                "similarity_with_answer": sim,
                "rerank_score": float(rerank_score),
                "document": doc.get("document"),
                "index": doc.get("index"),
                "text": text,
            })

    item["ground_truth_chunk_ids"] = gt_ids
    item["ground_truth_chunks"] = gt_chunks
    return item


def main():
    # 데이터 로드
    if not os.path.exists(INPUT_QA_PATH):
        raise FileNotFoundError(f"❌ 입력 파일 없음: {INPUT_QA_PATH}")

    print(f"📂 QA 데이터 로드: {INPUT_QA_PATH}")
    data = load_qa_dataset(INPUT_QA_PATH)
    print(f"✅ 총 {len(data)}개 샘플")

    # 임베딩 모델 로드
    print("📦 임베딩 모델 로딩 중...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # reranker 로드
    print("📦 BGE reranker 로딩 중...")
    reranker_pipeline = load_reranker()

    processed = []
    num_with_gt = 0

    for item in tqdm(data, desc="Ground Truth 생성 중"):
        new_item = build_ground_truth_for_item(item, embedder, reranker_pipeline)
        if new_item.get("ground_truth_chunk_ids"):
            num_with_gt += 1
        processed.append(new_item)

    os.makedirs(os.path.dirname(OUTPUT_QA_PATH), exist_ok=True)
    with open(OUTPUT_QA_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print("\n🎉 완료!")
    print(f"총 샘플: {len(processed)}")
    print(f"Ground Truth 생성된 샘플: {num_with_gt}")
    print(f"💾 저장 위치: {OUTPUT_QA_PATH}")


if __name__ == "__main__":
    main()
