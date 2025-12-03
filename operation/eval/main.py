# operation/eval/main.py
import mlflow
import uuid
import logging
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer  # (현재 직접 사용 X, 필요 없으면 삭제해도 됨)

from utils.models import models
from utils.generate_answer import generate_answer
from utils.sentence_score import compute_semantic_similarity
from utils.perplexity import compute_perplexity
from utils.data_loader import load_and_shuffle_qa
from utils.inference_eval import EvalInferencer
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

        keyword_overlap = (
            len(answer_words.intersection(context_words)) / len(answer_words)
            if answer_words else 0.0
        )
        return (semantic_similarity * 0.7) + (keyword_overlap * 0.3)
    except Exception as e:
        logger.warning(f"Faithfulness 계산 실패: {e}")
        return 0.0


def compute_context_recall(context: str, reference: str) -> float:
    try:
        ref_tokens = set(reference.lower().split())
        ctx_tokens = set(context.lower().split())

        if not ref_tokens:
            return 0.0

        overlap = len(ref_tokens.intersection(ctx_tokens))
        recall = overlap / len(ref_tokens)
        return float(recall)
    except Exception as e:
        logger.warning(f"Context Recall 계산 실패: {e}")
        return 0.0


def compute_context_precision(context: str, reference: str) -> float:
    try:
        ref_tokens = set(reference.lower().split())
        ctx_tokens = set(context.lower().split())

        if not ctx_tokens:
            return 0.0

        overlap = len(ref_tokens.intersection(ctx_tokens))
        precision = overlap / len(ctx_tokens)
        return float(precision)
    except Exception as e:
        logger.warning(f"Context Precision 계산 실패: {e}")
        return 0.0


def compute_stability_metrics(latencies):

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
        r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\data\instrcution\sample_20.json"
    )
    logger.info(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")

    # 평가용 인퍼런서 초기화 (top_k/토큰 예산은 필요시 변경)
    inferencer = EvalInferencer(top_k=5, per_ref_tokens=256, total_ctx_tokens=1024)

    run_id = str(uuid.uuid4())[:8]
    today = datetime.today().strftime("%Y%m%d")

    # baseline / tuned 우선 사용, 없으면 models 전체
    candidate_names = ["baseline", "tuned"]
    model_names = [m for m in candidate_names if m in models] or list(models.keys())

    for model_name in model_names:
        run_name = f"{model_name}_Hard_Metrics_{run_id}_{today}"
        logger.info(f"[{model_name}] 실험 Run 시작: {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("evaluation_type", "hard_metrics_rag_aware")
            mlflow.log_param("test_samples", len(test_data))
            use_rag = bool(models.get(model_name, {}).get("use_rag", True))
            mlflow.log_param("use_rag", use_rag)

            semantic_scores = []
            perplexities = []
            precision_at_k_scores = []
            faithfulness_scores = []
            latencies = []
            retrieval_ctx_sims = []
            context_recalls = []
            context_precisions = []

            logger.info(f"[{model_name}] 평가 시작... (use_rag={use_rag})")

            for i, item in enumerate(test_data):
                if i % 10 == 0:
                    logger.info(
                        f"[{model_name}] 진행률: {i}/{len(test_data)} "
                        f"({i / len(test_data) * 100:.1f}%)"
                    )

                question = item.get("question", item.get("QUESTION", "")) or ""
                reference = item.get("answer", item.get("ANSWER", "")) or ""

                try:
                    out = inferencer.infer(question, model_name=model_name, use_rag=use_rag)
                    generated_answer = out["answer"]
                    latency = out["latency_sec"]
                    latencies.append(latency)
                    semantic_sim = compute_semantic_similarity(generated_answer, reference)
                    perplexity = compute_perplexity(generated_answer, model_name=model_name)
                    precision_k = compute_precision_at_k(
                        question, generated_answer, reference, k=5
                    )


                    source_ctx = out.get("context", "") if use_rag else reference
                    faithfulness = compute_faithfulness(
                        generated_answer,
                        source_context=source_ctx,
                        reference_answer=reference
                    )

                    # 저장
                    semantic_scores.append(semantic_sim)
                    perplexities.append(perplexity)
                    precision_at_k_scores.append(precision_k)
                    faithfulness_scores.append(faithfulness)

                    if use_rag:
                        try:
                            if not hasattr(main, "_ctx_embed"):
                                main._ctx_embed = SentenceTransformer('all-MiniLM-L6-v2')
                            emb = main._ctx_embed
                            c = emb.encode([source_ctx])
                            r = emb.encode([reference])
                            ctx_sim = float(cosine_similarity(c, r)[0][0])
                            retrieval_ctx_sims.append(ctx_sim)
                        except Exception:
                            pass

                        # (2) Context Recall / Precision
                        try:
                            recall = compute_context_recall(source_ctx, reference)
                            precision = compute_context_precision(source_ctx, reference)
                            context_recalls.append(recall)
                            context_precisions.append(precision)
                        except Exception:
                            pass

                    # 중간 로그
                    if (i + 1) % 50 == 0:
                        msg = (
                            f"[{model_name}] #{i + 1}: "
                            f"Lat={latency:.3f}s, "
                            f"P@5={precision_k:.3f}, "
                            f"Faith={faithfulness:.3f}"
                        )
                        if use_rag and retrieval_ctx_sims:
                            msg += f", R-ctx-sim={retrieval_ctx_sims[-1]:.3f}"
                        logger.info(msg)

                except Exception as e:
                    logger.error(f"[{model_name}] 샘플 {i} 처리 실패: {e}")
                    continue

            # ------------------------
            # 최종 메트릭 집계
            # ------------------------
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
            mlflow.log_metric(
                "latency_cv",
                (std_latency / avg_latency) if avg_latency > 0 else 0.0
            )
            mlflow.log_metric("samples_processed", len(semantic_scores))

            if use_rag and retrieval_ctx_sims:
                mlflow.log_metric(
                    "avg_retrieval_context_sim",
                    float(np.mean(retrieval_ctx_sims))
                )

            if use_rag and context_recalls and context_precisions:
                mlflow.log_metric(
                    "avg_context_recall",
                    float(np.mean(context_recalls))
                )
                mlflow.log_metric(
                    "avg_context_precision",
                    float(np.mean(context_precisions))
                )

            # ------------------------
            # 결과 로그 출력
            # ------------------------
            logger.info(f"[{model_name}] === 최종 평가 결과 ===")
            logger.info(f"     기존 메트릭:")
            logger.info(f"    - Semantic Similarity: {avg_semantic_sim:.4f}")
            logger.info(f"    - Perplexity: {avg_perplexity:.4f}")
            logger.info(f"      하드 지표:")
            logger.info(f"    - Precision@5: {avg_precision_at_k:.4f}")
            logger.info(f"    - Faithfulness: {avg_faithfulness:.4f}")
            logger.info(f"     지연시간 메트릭:")
            logger.info(f"    - 평균 지연시간: {avg_latency:.3f}초")
            logger.info(f"    - 표준편차: {std_latency:.3f}초")
            logger.info(f"    - P95 지연시간: {p95_latency:.3f}초")
            logger.info(
                f"    - 변동계수: "
                f"{(std_latency / avg_latency) if avg_latency > 0 else 0:.3f}"
            )

            if use_rag:
                logger.info(f" RAG 보조지표:")
                if retrieval_ctx_sims:
                    logger.info(
                        f"    - Retrieval Ctx ~ Ref 유사도: "
                        f"{float(np.mean(retrieval_ctx_sims)):.4f}"
                    )
                if context_recalls and context_precisions:
                    logger.info(
                        f"    - Context Recall: "
                        f"{float(np.mean(context_recalls)):.4f}"
                    )
                    logger.info(
                        f"    - Context Precision: "
                        f"{float(np.mean(context_precisions)):.4f}"
                    )

    logger.info("=== 하드 지표 기반 모델 평가 완료 ===")


if __name__ == "__main__":
    main()
