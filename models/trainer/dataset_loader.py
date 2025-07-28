from datasets import load_dataset

def load_json_dataset(json_path: str, split="train"):
    return load_dataset("json", data_files=json_path)[split]

def preprocess_dataset(dataset, tokenizer, max_length=1024):
    def format_conversations(example):
        # conversations 형태로 변환
        conversations = [
            {"role": "user", "content": example["QUESTION"]},
            {"role": "assistant", "content": example["ANSWER"]}
        ]
        
        # tokenizer의 chat template 적용해서 텍스트 생성
        formatted_text = tokenizer.apply_chat_template(
            conversations, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}
    
    def tokenize(example):
        tokenized = tokenizer(
            example["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        # labels 추가 (input_ids와 동일하게 설정)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 대화 형태로 포맷팅 후 토크나이징
    formatted_dataset = dataset.map(format_conversations)
    return formatted_dataset.map(tokenize, batched=True)