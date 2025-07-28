import torch
import math
from .models import models

@torch.no_grad()
def compute_perplexity(text: str, model_name: str = "baseline", max_length: int = 2048) -> float:
    model = models[model_name]["model"]
    tokenizer = models[model_name]["tokenizer"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    return math.exp(loss.item())