from datetime import datetime
import uuid

mlflow_config = {
    "tracking_uri": "http://localhost:5001",
    "experiment_name": "model-finetuning",
    "run_name": f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
}

model_path = {
    "path": "unsloth/gemma-3-4b-it",
}

training_config = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1.5,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 250,
    "save_total_limit": 2,
    "bf16": True,
    "logging_dir": "./logs",
    "report_to": "mlflow",
    "run_name": mlflow_config["run_name"],
    "save_safetensors": False,
}