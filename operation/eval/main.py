import mlflow
import uuid
from datetime import datetime
from utils.models import models
from utils.sentence_score import similarity_model, compute_semantic_similarity
from utils.perplexity import compute_perplexity
from utils.data_loader import load_and_shuffle_qa

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("model_evaluation-A/B.Test")

    test_data = load_and_shuffle_qa("C:/Users/dm_ohminchan/Model/data/instrcution/generation_QA_set_20250722.json")

    for model_name in ["baseline", "tuned"]:
        model = models[model_name]["model"]
        tokenizer = models[model_name]["tokenizer"]

        semantic_scores = []
        perplexities = []

        run_id = str(uuid.uuid4())[:8]
        today = datetime.today().strftime("%Y%m%d")
        run_name = f"{model_name}_{run_id}_{today}"

        with mlflow.start_run(run_name=run_name):
            for item in test_data:
                # inference 생략된 상태
                generated_answer = item["answer"]  # 실제론 모델로 생성해야 함

                sim = compute_semantic_similarity(generated_answer, item["answer"])
                ppl = compute_perplexity(model, tokenizer, generated_answer)

                semantic_scores.append(sim)
                perplexities.append(ppl)

            mlflow.log_metric("avg_semantic_similarity", sum(semantic_scores) / len(semantic_scores))
            mlflow.log_metric("avg_perplexity", sum(perplexities) / len(perplexities))

if __name__ == "__main__":
    main()
