from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    return model, tokenizer
