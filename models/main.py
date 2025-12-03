from transformers import Trainer
import logging

from config.configs import mlflow_config, model_path, training_config

import mlflow

from trainer.model_loader import load_model_and_tokenizer
from trainer.arguments import create_training_args
from trainer.dataset_loader import load_json_dataset, preprocess_dataset

import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("파인튜닝 시작")
    
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    logger.info("MLflow 설정 완료")

    logger.info("모델 및 토크나이저 로드 시작...")
    model, tokenizer = load_model_and_tokenizer(model_path["path"])
    logger.info(f"모델 로드 완료: {model_path['path']}")

    logger.info("데이터셋 로드 시작...")
    dataset = load_json_dataset(json_path="C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/data/instrcution/qa_with_perspectives_cleaned_filled.json")
    logger.info(f"데이터셋 로드 완료: {len(dataset)}개 샘플")
    
    logger.info("데이터 전처리 시작...")
    train_dataset = preprocess_dataset(dataset=dataset, tokenizer=tokenizer)
    logger.info("데이터 전처리 완료")

    training_args = create_training_args(config=training_config)
    logger.info("훈련 매개변수 설정 완료")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    logger.info("Trainer 초기화 완료")

    with mlflow.start_run():
        logger.info("훈련 시작!")
        trainer.train()
        logger.info("훈련 완료")

        logger.info("모델 MLflow에 로깅 시작...")
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="outputs",
            task="text-generation",
            registered_model_name=mlflow_config["experiment_name"]
        )
        logger.info("모델 로깅 완료")
    
    logger.info("파인튜닝 훈련 완료")

if __name__ == "__main__":
    main()