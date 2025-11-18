# operation/eval/utils/inference_eval.py
from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer

from .faiss_test import vector_search
from .models import models
from .generate_answer import generate_answer

# 전역 토크나이저 (컨텍스트 토큰 길이 계산/클램핑에만 사용)
_tok = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it", trust_remote_code=True)

def _ids(text: str) -> List[int]:
    """문자열 → 토큰 ID 리스트 (special tokens 미포함)"""
    return _tok.encode(text or "", add_special_tokens=False)

def _clamp_by_tokens(texts: List[str], per_ref_tokens: int, total_ctx_tokens: int) -> str:
    """
    여러 개의 reference 텍스트를 토큰 예산에 맞춰 잘라 붙인다.
      - 각 ref는 per_ref_tokens로 개별 트렁케이션
      - 전체 누적은 total_ctx_tokens를 넘지 않도록 누적
    """
    out, used = [], 0
    for t in texts:
        ids = _ids(t)
        if len(ids) > per_ref_tokens:
            t = _tok.decode(ids[:per_ref_tokens])
            ids = ids[:per_ref_tokens]
        if used + len(ids) > total_ctx_tokens:
            break
        out.append(t)
        used += len(ids)
    return "\n\n".join(out)

def build_context(query: str, top_k: int = 5, per_ref_tokens: int = 256, total_ctx_tokens: int = 1024) -> Tuple[str, List[Dict[str, Any]]]:
    """
    로컬 FAISS로 검색 → 텍스트만 추출 → 토큰 예산으로 클램프
    반환: (컨텍스트 문자열, 원본 히트 목록)
    """
    hits = vector_search(query, top_k=top_k)
    texts = [h.get("text", "") for h in hits if h.get("text")]
    context = _clamp_by_tokens(texts, per_ref_tokens=per_ref_tokens, total_ctx_tokens=total_ctx_tokens)[:8000]
    return context, hits

class EvalInferencer:
    """
    평가 전용 인퍼런서
      - use_rag=True  : 로컬 FAISS 검색 → 컨텍스트 주입 → 답변 생성
      - use_rag=False : 컨텍스트 없이 순수 LLM 프롬프트로 답변 생성
    """
    def __init__(self, top_k: int = 5, per_ref_tokens: int = 256, total_ctx_tokens: int = 1024):
        self.top_k = top_k
        self.per_ref_tokens = per_ref_tokens
        self.total_ctx_tokens = total_ctx_tokens

    def infer(self, question: str, model_name: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        한 샘플에 대한 생성 수행 + 지연시간(ms)
        반환: answer, latency_sec, use_rag, context, references
        """
        entry = models[model_name]
        model, tokenizer = entry["model"], entry["tokenizer"]

        # 1) 컨텍스트 구성 (RAG On일 때만)
        if use_rag:
            ctx, refs = build_context(
                question,
                top_k=self.top_k,
                per_ref_tokens=self.per_ref_tokens,
                total_ctx_tokens=self.total_ctx_tokens,
            )
            prompt = (
                "You are a helpful assistant. Use ONLY the provided context to answer.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )
        else:
            refs, ctx = [], ""
            prompt = (
                "Answer the question concisely.\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )

        # 2) 생성 + 지연시간
        t0 = time.time()
        answer = generate_answer(prompt, model, tokenizer)
        latency = time.time() - t0

        return {
            "answer": answer,
            "latency_sec": latency,
            "use_rag": use_rag,
            "context": ctx,
            "references": refs,
        }
