from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
torch._dynamo.config.disable = True

# 평가용 유사도 모델
similarity_model = SentenceTransformer("dragonkue/bge-reranker-v2-m3-ko")

models = {
    "baseline": {
        "path": "unsloth/gemma-3-4b-it",
        "tokenizer": None,
        "model": None,
    },
    "tuned": {
        "path": "C:/Users/dm_ohminchan/RAGLLM-Feature-model-train/models/checkpoint-2352",
        "tokenizer": None,
        "model": None,
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"

for name, config in models.items():
    config["tokenizer"] = AutoTokenizer.from_pretrained(config["path"], trust_remote_code=True)
    config["model"] = AutoModelForCausalLM.from_pretrained(
        config["path"], 
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
