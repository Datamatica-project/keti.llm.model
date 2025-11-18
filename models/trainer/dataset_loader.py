from datasets import load_dataset

def load_json_dataset(json_path: str, split="train"):
    return load_dataset("json", data_files=json_path)[split]

def format_conversations(example, tokenizer):
    chat = [
        {"role": "user", "content": example["QUESTION"]},
        {"role": "assistant", "content": example["ANSWER"]}
    ]
    formatted_text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_text}

def tokenize_conversations(example, tokenizer, max_length=1024):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def preprocess_dataset(dataset, tokenizer, max_length=1024):
    # 1단계: chat template 적용
    formatted_dataset = dataset.map(lambda x: format_conversations(x, tokenizer), remove_columns=dataset.column_names)

    # 2단계: 토크나이즈 및 레이블 생성
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_conversations(x, tokenizer, max_length),
        batched=False
    )
    return tokenized_dataset

