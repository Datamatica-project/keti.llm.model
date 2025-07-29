import mlflow
import uuid
import logging
from datetime import datetime

from utils.models import models
from utils.generate_answer import generate_answer
from utils.sentence_score import compute_semantic_similarity
from utils.perplexity import compute_perplexity
from utils.data_loader import load_and_shuffle_qa

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

def main():
    logger.info("모델 평가 시작")

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("model_evaluation-A/B.Test")
    logger.info("MLflow 설정 완료")

    test_data = load_and_shuffle_qa("C:/Users/dm_ohminchan/Model/data/instrcution/generation_QA_set_20250722.json")
    logger.info(f"테스트 데이터 로드 완료: {len(test_data)}개 샘플")

    run_id = str(uuid.uuid4())[:8]
    today = datetime.today().strftime("%Y%m%d")

    for model_name in ["baseline", "tuned"]:
        run_name = f"{model_name}_AB_test_{run_id}_{today}"
        logger.info(f"[{model_name}] 실험 Run 시작: {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)

            model = models[model_name]["model"]
            tokenizer = models[model_name]["tokenizer"]

            semantic_scores = []
            perplexities = []

            for i, item in enumerate(test_data):
                if i % 10 == 0:
                    logger.info(f"{model_name}: {i}/{len(test_data)} 진행중")

                question = item["question"]
                reference = item["answer"]


                generated_answer = generate_answer(question, model, tokenizer)

                sim = compute_semantic_similarity(generated_answer, reference)
                ppl = compute_perplexity(generated_answer, model_name=model_name)

                semantic_scores.append(sim)
                perplexities.append(ppl)

            avg_sim = sum(semantic_scores) / len(semantic_scores)
            avg_ppl = sum(perplexities) / len(perplexities)

            mlflow.log_metric("avg_semantic_similarity", avg_sim)
            mlflow.log_metric("avg_perplexity", avg_ppl)

            logger.info(f"[{model_name}] 결과 - Similarity: {avg_sim:.4f}, Perplexity: {avg_ppl:.4f}")

    logger.info("=== 모델 평가 완료 ===")

if __name__ == "__main__":
    main()
