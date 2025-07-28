import json
import random
from typing import List, Dict

def load_and_shuffle_qa(path: str, seed: int = 42, sample_size: int = 100) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 질문-답변 형식만 필터링
    qa_pairs = [
        {"question": item["QUESTION"], "answer": item["ANSWER"]}
        for item in data if "QUESTION" in item and "ANSWER" in item
    ]

    random.seed(seed)
    random.shuffle(qa_pairs)

    return qa_pairs[:sample_size]
