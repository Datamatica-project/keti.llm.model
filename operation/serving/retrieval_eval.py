# operation/serving/retrieval_eval.py (예시 위치)

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Set

from tqdm import tqdm

from .search import vector_search                 # RAG 검색 함수 :contentReference[oaicite:0]{index=0}
from .reranker import load_reranker, rerank_with_bge  # BGE reranker :contentReference[oaicite:1]{index=1}


# ===== 설정 =====
# ✅ Ground Truth가 포함된 QA 파일 (앞에서 만든 OUTPUT_QA_PATH)
QA_WITH_GT_PATH = "C:/Users/dm_ohminchan/Model/data/instrcution/generation_QA_with_retrieval_gt.json"

# ✅ Retrieval / Rerank 설정 (평가 시 사용할 Top-K)
RETRIEVAL_TOP_K = 20    # vector_search에서 가져올 개수
RERANK_TOP_K = 8        # rerank 후 최종 평가에 쓸 개수


# ===== 유틸 =====

def load_qa_with_gt(path: str) -> List[Dict[str, Any]]:
    """Ground Truth 정보가 포함된 QA JSON 로드"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"QA 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 리스트 안에 또 리스트로 들어있는 경우 평탄화
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for sub in data:
            flat.extend(sub)
        data = flat

    return data


def extract_question(item: Dict[str, Any]) -> str:
    """question / QUESTION 둘 다 지원"""
    return (item.get("question") or item.get("QUESTION") or "").strip()


def extract_gt_ids(item: Dict[str, Any]) -> List[str]:
    """ground_truth_chunk_ids 필드에서 GT ID 리스트 가져오기"""
    gt_ids = item.get("ground_truth_chunk_ids") or []
    # 혹시 문자열로 들어가 있으면 리스트로 감싸기
    if isinstance(gt_ids, str):
        gt_ids = [gt_ids]
    return gt_ids


def compute_recall_precision(
    gt_ids: Set[str],
    retrieved_ids: List[str]
) -> (float, float):
    """한 샘플에 대해 Context Recall / Precision 계산"""

    if not gt_ids:
        return 0.0, 0.0

    retrieved_set = set(retrieved_ids)
    hit = gt_ids.intersection(retrieved_set)

    recall = len(hit) / len(gt_ids) if len(gt_ids) > 0 else 0.0
    precision = len(hit) / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0

    return recall, precision


def main():
    print(f"📂 QA + Ground Truth 데이터 로드: {QA_WITH_GT_PATH}")
    data = load_qa_with_gt(QA_WITH_GT_PATH)
    print(f"✅ 총 {len(data)}개 샘플 로드")

    print("📦 BGE reranker 로드 중...")
    reranker_pipeline = load_reranker()

    recalls: List[float] = []
    precisions: List[float] = []

    total_samples = 0
    evaluated_samples = 0

    for item in tqdm(data, desc="Context Recall / Precision 계산 중"):
        total_samples += 1

        question = extract_question(item)
        gt_ids_list = extract_gt_ids(item)
        gt_ids = set(gt_ids_list)

        # GT가 없는 샘플은 평가에서 제외
        if not question or not gt_ids:
            continue

        try:
            # 1) vector_search로 후보 문서 가져오기 :contentReference[oaicite:2]{index=2}
            vector_results = vector_search(question, top_k=RETRIEVAL_TOP_K)
            if not vector_results:
                continue

            # 2) BGE rerank 후 상위 K개만 사용 :contentReference[oaicite:3]{index=3}
            reranked = rerank_with_bge(
                query=question,
                docs=vector_results,
                reranker_pipeline=reranker_pipeline,
                top_k=RERANK_TOP_K
            )
            if not reranked:
                continue

            # 3) reranked 결과에서 chunk_id 추출
            retrieved_ids: List[str] = []
            for doc, score in reranked:
                chunk_id = (
                    doc.get("chunk_id")
                    or f"{doc.get('document')}::{doc.get('index')}"
                )
                retrieved_ids.append(str(chunk_id))

            # 4) 해당 샘플에 대한 Recall / Precision 계산
            recall, precision = compute_recall_precision(gt_ids, retrieved_ids)

            recalls.append(recall)
            precisions.append(precision)
            evaluated_samples += 1

        except Exception as e:
            print(f"[WARN] 샘플 처리 실패: {e}")
            continue

    if evaluated_samples == 0:
        print("⚠ 평가 가능한 샘플이 없습니다. Ground Truth가 비어있는지 확인하세요.")
        return

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)

    print("\n=== 📊 RAG Retrieval 평가 결과 ===")
    print(f"총 샘플 수: {total_samples}")
    print(f"평가에 사용된 샘플 수 (GT 존재): {evaluated_samples}")
    print(f"Context Recall (평균): {avg_recall:.4f}")
    print(f"Context Precision (평균): {avg_precision:.4f}")

    # 간단히 통계 몇 개 더 출력 (원하면 나중에 MLflow로 로깅해도 됨)
    print(f"Recall ≥ 0.7 비율: {sum(1 for r in recalls if r >= 0.7) / len(recalls):.3f}")
    print(f"Precision ≥ 0.6 비율: {sum(1 for p in precisions if p >= 0.6) / len(precisions):.3f}")


if __name__ == "__main__":
    main()
