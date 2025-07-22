from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Tuple


def load_reranker(model_name: str = "dragonkue/bge-reranker-v2-m3-ko"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,  # 점수 하나만 반환
<<<<<<< HEAD
=======
        max_length=8192,
>>>>>>> 0c89ddde410f1f3f2d72ee223ac04c56dff24b2e
        device=0
    )


# Rerank 함수
def rerank_with_bge(
    query: str,
    docs: List[Dict[str, str]],
    reranker_pipeline,
    top_k: int = 5
) -> List[Tuple[Dict[str, str], float]]:

    # 입력 포맷: "질문 [SEP] 문서"
    pairs = [f"{query} [SEP] {doc['text']}" for doc in docs]
    scores = reranker_pipeline(pairs)

    # 문서와 점수 묶기
    ranked = sorted(
        zip(docs, [score['score'] for score in scores]),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]
