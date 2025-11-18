# operation/eval/main.py
import mlflow
import uuid
import logging
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.models import models
from utils.generate_answer import generate_answer  # 남겨둠 (inferencer 내부에서 사용)
from utils.sentence_score import compute_semantic_similarity
from utils.perplexity import compute_perplexity
from utils.data_loader import load_and_shuffle_qa
from utils.inference_eval import EvalInferencer  # ★ 추가: 평가 전용 인퍼런서

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------
# 하드 지표 계산 함수들
# ------------------------
def compute_precision_at_k(question, generated_answer, reference_answer, k=5):
    """
    Precision@k: 생성된 답변에서 상위 k개 핵심 정보의 정확도
    """
    try:
        reference_keywords = set(reference_answer.lower().split())
        generated_keywords = set(generated_answer.lower().split())
        top_k_generated = sorted(generated_keywords, key=len, reverse=True)[:k]
        relevant_count = sum(1 for keyword in top_k_generated if keyword in reference_keywords)
        return (relevant_count / k) if k > 0 else 0.0
    except Exception as e:
        logger.warning(f"Precision@k 계산 실패: {e}")
        return 0.0


def compute_faithfulness(generated_answer, source_context="", reference_answer=""):

    try:
        if not source_context:
            source_context = reference_answer

        if not hasattr(compute_faithfulness, 'model'):
            compute_faithfulness.model = SentenceTransformer('all-MiniLM-L6-v2')

        model = compute_faithfulness.model

        answer_embedding = model.encode([generated_answer])
        context_embedding = model.encode([source_context])
        semantic_similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]

        answer_words = set(generated_answer.lower().split())
        context_words = set(source_context.lower().split())

        keyword_overlap = (len(answer_words.intersection(context_words)) / len(answer_words)) if answer_words else 0.0
        return (semantic_similarity * 0.7) + (keyword_overlap * 0.3)
    except Exception as e:
        logger.warning(f"Faithfulness 계산 실패: {e}")
        return 0.0


def compute_stability_metrics(latencies):
    """
    지연시간 안정성 메트릭 계산
    """
    if len(latencies) == 0:
        return 0.0, 0.0, 0.0

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    return avg_latency, std_latency, p95_latency


def main():
    logger.info("=== 하드 지표 기반 모델 평가 시작 (RAG vs LLM) ===")

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("model_evaluation-RAG-vs-LLM-Hard-Metrics")
    logger.info("MLflow 설정 완료")

    test_data = load_and_shuffle_qa(
        r"C:\Users\dm_ohminchan\Model\data\instrcution\generation_QA_set_20250722_251029a.json"
    )
    logger.info(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")

    # ★ 평가용 인퍼런서 초기화 (top_k/토큰 예산은 필요시 변경)
    inferencer = EvalInferencer(top_k=5, per_ref_tokens=256, total_ctx_tokens=1024)

    run_id = str(uuid.uuid4())[:8]
    today = datetime.today().strftime("%Y%m%d")

    # 기존 리스트가 있으면 우선 사용, 없으면 models의 키 전체 사용
    candidate_names = ["baseline", "tuned"]
    model_names = [m for m in candidate_names if m in models] or list(models.keys())

    for model_name in model_names:
        run_name = f"{model_name}_Hard_Metrics_{run_id}_{today}"
        logger.info(f"[{model_name}] 실험 Run 시작: {run_name}")

        with mlflow.start_run(run_name=run_name):
            # 파라미터 로깅
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("evaluation_type", "hard_metrics_rag_aware")
            mlflow.log_param("test_samples", len(test_data))

            # RAG on/off 플래그 읽기 (없으면 False)
            use_rag = bool(models.get(model_name, {}).get("use_rag", False))
            mlflow.log_param("use_rag", use_rag)

            # 모델 핸들 확보(실제 생성은 inferencer가 내부 generate_answer에서 사용)
            # model = models[model_name]["model"]
            # tokenizer = models[model_name]["tokenizer"]

            # 메트릭 저장용 리스트
            semantic_scores = []
            perplexities = []
            precision_at_k_scores = []
            faithfulness_scores = []
            latencies = []
            retrieval_ctx_sims = []  # RAG 전용 보조지표

            logger.info(f"[{model_name}] 평가 시작... (use_rag={use_rag})")

            for i, item in enumerate(test_data):
                if i % 10 == 0:
                    logger.info(f"[{model_name}] 진행률: {i}/{len(test_data)} ({i / len(test_data) * 100:.1f}%)")

                question = item.get("question", item.get("QUESTION", "")) or ""
                reference = item.get("answer", item.get("ANSWER", "")) or ""

                try:
                    #  1) 생성 + 지연시간 (RAG/LLM)
                    out = inferencer.infer(question, model_name=model_name, use_rag=use_rag)
                    generated_answer = out["answer"]
                    latency = out["latency_sec"]
                    latencies.append(latency)

                    #  2) 기존 메트릭
                    semantic_sim = compute_semantic_similarity(generated_answer, reference)
                    perplexity = compute_perplexity(generated_answer, model_name=model_name)

                    #  3) 하드 지표
                    precision_k = compute_precision_at_k(question, generated_answer, reference, k=5)

                    # 4) Faithfulness: RAG이면 retrieved context, 아니면 reference
                    source_ctx = out.get("context", "") if use_rag else reference
                    faithfulness = compute_faithfulness(generated_answer, source_context=source_ctx, reference_answer=reference)

                    # 저장
                    semantic_scores.append(semantic_sim)
                    perplexities.append(perplexity)
                    precision_at_k_scores.append(precision_k)
                    faithfulness_scores.append(faithfulness)

                    # (선택) RAG 전용 보조지표: retrieved context vs reference 의미 유사도
                    if use_rag:
                        try:
                            if not hasattr(main, "_ctx_embed"):
                                main._ctx_embed = SentenceTransformer('all-MiniLM-L6-v2')
                            emb = main._ctx_embed
                            c = emb.encode([source_ctx])
                            r = emb.encode([reference])
                            ctx_sim = float(cosine_similarity(c, r)[0][0])
                            retrieval_ctx_sims.append(ctx_sim)
                        except Exception as _:
                            pass

                    # 중간 로그
                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"[{model_name}] #{i + 1}: Lat={latency:.3f}s, "
                            f"P@5={precision_k:.3f}, Faith={faithfulness:.3f}"
                            + (f", R-ctx-sim={retrieval_ctx_sims[-1]:.3f}" if use_rag and retrieval_ctx_sims else "")
                        )

                except Exception as e:
                    logger.error(f"[{model_name}] 샘플 {i} 처리 실패: {e}")
                    continue

            # 최종 메트릭
            avg_semantic_sim = float(np.mean(semantic_scores)) if semantic_scores else 0.0
            avg_perplexity = float(np.mean(perplexities)) if perplexities else 0.0
            avg_precision_at_k = float(np.mean(precision_at_k_scores)) if precision_at_k_scores else 0.0
            avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

            avg_latency, std_latency, p95_latency = compute_stability_metrics(latencies)

            # MLflow 로깅
            mlflow.log_metric("avg_semantic_similarity", avg_semantic_sim)
            mlflow.log_metric("avg_perplexity", avg_perplexity)
            mlflow.log_metric("avg_precision_at_5", avg_precision_at_k)
            mlflow.log_metric("avg_faithfulness", avg_faithfulness)
            mlflow.log_metric("avg_latency_seconds", avg_latency)
            mlflow.log_metric("std_latency_seconds", std_latency)
            mlflow.log_metric("p95_latency_seconds", p95_latency)
            mlflow.log_metric("latency_cv", (std_latency / avg_latency) if avg_latency > 0 else 0.0)
            mlflow.log_metric("samples_processed", len(semantic_scores))

            if use_rag and retrieval_ctx_sims:
                mlflow.log_metric("avg_retrieval_context_sim", float(np.mean(retrieval_ctx_sims)))

            # 결과 로그
            logger.info(f"[{model_name}] === 최종 평가 결과 ===")
            logger.info(f"     기존 메트릭:")
            logger.info(f"    - Semantic Similarity: {avg_semantic_sim:.4f}")
            logger.info(f"    - Perplexity: {avg_perplexity:.4f}")
            logger.info(f"      하드 지표:")
            logger.info(f"    - Precision@5: {avg_precision_at_k:.4f}")
            logger.info(f"    - Faithfulness: {avg_faithfulness:.4f}")
            logger.info(f"  ⚡ 지연시간 메트릭:")
            logger.info(f"    - 평균 지연시간: {avg_latency:.3f}초")
            logger.info(f"    - 표준편차: {std_latency:.3f}초")
            logger.info(f"    - P95 지연시간: {p95_latency:.3f}초")
            logger.info(f"    - 변동계수: {(std_latency / avg_latency) if avg_latency > 0 else 0:.3f}")
            if use_rag and retrieval_ctx_sims:
                logger.info(f"  🔎 RAG 보조지표:")
                logger.info(f"    - Retrieval Ctx ~ Ref 유사도: {float(np.mean(retrieval_ctx_sims)):.4f}")

    logger.info("=== 하드 지표 기반 모델 평가 완료 ===")


if __name__ == "__main__":
    main()
