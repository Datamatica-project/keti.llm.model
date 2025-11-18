from transformers import TrainingArguments

def create_training_args(config: dict) -> TrainingArguments:
    return TrainingArguments(**config)