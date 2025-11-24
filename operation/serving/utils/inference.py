from __future__ import annotations

from .reranker import load_reranker, rerank_with_bge
from .search import vector_search
from dto.routings import RoutingResult, RouteType

from .buffer import save_session_memory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer

from typing import Dict, Any, List
import os
import re
import numpy as np
from dotenv import load_dotenv

# .env 로드 (프로젝트 루트에 있는 .env를 읽음)
load_dotenv()

# LangSmith 설정: 민감한 값(API 키)은 .env / 환경변수에서만 관리
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "Keti"))

reranker = load_reranker()
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it", trust_remote_code=True)

SMALLTALK = {"하이", "안녕", "안녕하세요", "ㅎㅇ", "hello", "hi", "hey", "hi!", "hello!"}
SMALLTALK_RE = re.compile(r"^[\s\W_]+$")


def is_smalltalk(q: str) -> bool:
    qn = re.sub(r"\s+", " ", q.strip().lower())
    return (len(qn) <= 1) or (qn in SMALLTALK) or (SMALLTALK_RE.match(qn) is not None)


def _encode_tokens(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _tok_len(text: str) -> int:
    return len(_encode_tokens(text))


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    ids = _encode_tokens(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def clamp_context_by_tokens(
    texts: List[str],
    per_ref_tokens: int = 256,
    total_tokens: int = 1024,
) -> str:
    out, used = [], 0
    for t in texts:
        t2 = _truncate_text_to_tokens(t, per_ref_tokens)
        n = len(_encode_tokens(t2))
        if used + n > total_tokens:
            break
        out.append(t2)
        used += n
    return "\n".join(out)


def count_tokens(messages: List) -> int:
    total_tokens = 0
    for message in messages:
        if hasattr(message, "content"):
            total_tokens += len(_encode_tokens(message.content))
        else:
            total_tokens += len(_encode_tokens(str(message)))
    return total_tokens


def trim_history_to_budget(
    prev_msgs: List[HumanMessage],
    curr_msgs: List[HumanMessage],
    max_ctx: int = 4096,
    gen_tokens: int = 300,
    buffer_tokens: int = 128,
) -> List[HumanMessage]:
    budget = max_ctx - gen_tokens - buffer_tokens
    msgs = prev_msgs + curr_msgs
    while count_tokens(msgs) > budget and prev_msgs:
        # 대화쌍 단위로 제거 시도(없으면 1개씩)
        if len(prev_msgs) >= 2:
            prev_msgs.pop(0)
            prev_msgs.pop(0)
        else:
            prev_msgs.pop(0)
        msgs = prev_msgs + curr_msgs
    return msgs


def route_query(query: str) -> RoutingResult:
    llm = ChatOpenAI(
        model_name="unsloth/gemma-3-4b-it",
        openai_api_base="http://vllm.api:8000/v1",
        max_tokens=30,
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "sk-fake-key"),
    )

    prompt = f"""[지침]
- 농업 기술, 작물 재배, 병해충, 농약, 작물 효능, 품종, 생육 환경, 재배 시기, 수확법 등 전문 정보가 필요한 질문이면 "농업검색"
- 단순 인사, 감정 표현, 일상 대화, 잡담은 "일반대화"

[예시]
질문: "마늘의 효능이 뭐야?"
답변: 농업검색

질문: "안녕하세요?"
답변: 일반대화

질문: "비료를 언제 줘야 하나요?"
답변: 농업검색

질문: "배추 재배법 알려줘"
답변: 농업검색

질문: "고추에 벌레가 생겼어요. 어떻게 하죠?"
답변: 농업검색

질문: "감사합니다!"
답변: 일반대화

[입력]
질문: "{query}"

답변:
"""
    try:
        response = llm.invoke(prompt)
        # 정확 일치로 1줄만 판단하여 라우팅 오탐 방지
        content = response.content.strip().splitlines()[0].strip()

        if content == "농업검색":
            route: RouteType = "document_search"
            reasoning = "농업 전문 지식 필요"
        else:
            route = "general_chat"
            reasoning = "일반 대화"

        return RoutingResult(route=route, reasoning=reasoning)

    except Exception as e:
        # 실패 시 안전하게 문서 검색으로 라우팅
        return RoutingResult(route="document_search", reasoning=f"라우팅 오류: {str(e)}")


def generate_response(query: str, session_id: str) -> Dict[str, Any]:
    # 세션 메모리 준비
    memory = save_session_memory(session_id, "redis://192.168.0.150:6379")
    if memory is None:
        return {"error": "메모리 초기화 실패"}

    input_query = len(tokenizer.encode(query))

    # 스몰토크면 여기서 즉시 종료 (RAG/LLM/메모리 업데이트 생략)
    if is_smalltalk(query):
        return {
            "answer": "안녕하세요! 무엇을 도와드릴까요? 😊",
            "input_tokens": input_query,
            "completion_tokens": 0,
            "references": "",
            "rank": [],
        }

    # 라우팅
    routing: RoutingResult = route_query(query)
    print(f"라우팅: {routing['route']} - {routing['reasoning']}")

    # 검색 / 재랭킹
    references: List[Dict[str, Any]]
    if routing["route"] == "document_search":
        print(f"농업 검색 수행: {query}")
        # 검색 8개
        vector_results = vector_search(query, top_k=8)

        # 재랭킹: 4개
        reranked = rerank_with_bge(query, vector_results, reranker, top_k=4)
        THRESH = 0.35
        reranked = [(doc, score) for doc, score in reranked if score >= THRESH]

        references = [
            {"document": doc.get("document"), "text": doc.get("text"), "score": score}
            for doc, score in reranked
        ]
    else:
        print(f"일반 대화: {query}")
        references = []

    # 컨텍스트 후보 (ref 텍스트)
    ref_texts_all = [ref["text"] for ref in references]

    # 기본값
    MAX_CTX = 4096
    BUFFER = 192  # 여유 버퍼
    GEN_TOKENS = 300  # 기본 생성 길이
    per_ref_tokens = 256  # ref당 최대 토큰
    total_ctx_tokens = 1024  # 컨텍스트 전체 한도
    max_refs = 4  # 시작은 4개
    min_refs = 2  # 예산 초과 시 2개까지 축소

    # (문서검색일 땐 히스토리 최소화가 안전)
    previous_messages = memory.load_memory_variables({}).get("chat_history", [])
    if routing["route"] == "document_search":
        previous_messages = previous_messages[-2:]  # 마지막 1쌍만 유지(없으면 0)

    def build_prompt(ctx: str) -> str:
        if not ctx:
            return f"""사용자의 질문에 간결하고 명확하게 답하세요.

[질문]
{query}

지침:
- 200자 이내.
- 사실과 일반 상식 범위에서만.
- 불확실하면 단정하지 말고 필요한 정보(작물/상황 등)만 요청.
- 문장이 자연스럽게 끝나도록 작성.
"""
        return f"""아래 문서 내용을 참고하여 질문에 답하세요.

[문서]
{ctx}

[질문]
{query}

지침:
- 200자 이내.
- 사실과 일반 상식 범위에서만.
- 불확실하면 단정하지 말고 필요한 정보(작물/상황 등)만 요청.
- 문장이 자연스럽게 끝나도록 작성.
"""

    # === 4 → 2로 감소하는 적응형 예산 가드 ===
    while True:
        # 현재 ref 개수로 컨텍스트 구성
        use_texts = ref_texts_all[:max_refs] if references else []
        context = clamp_context_by_tokens(
            use_texts,
            per_ref_tokens=per_ref_tokens,
            total_tokens=total_ctx_tokens,
        )

        system_prompt = "당신은 간결하고 정확한 농업 상담가입니다. 주어진 지침을 엄격히 따르세요."
        prompt = build_prompt(context)
        current_messages = [HumanMessage(content=system_prompt + "\n\n" + prompt)]
        all_messages = previous_messages + current_messages

        msg_tokens = count_tokens(all_messages)
        total_requested = msg_tokens + GEN_TOKENS + BUFFER
        print(
            f"[BUDGET] refs={max_refs}, per_ref={per_ref_tokens}, ctx_total={total_ctx_tokens} | "
            f"msgs={msg_tokens}, gen={GEN_TOKENS}, total={total_requested}/{MAX_CTX}"
        )

        if total_requested <= MAX_CTX:
            # 예산 충족 → 루프 종료
            break

        # 1) 먼저 ref 개수를 4 → 2로 줄인다
        if max_refs > min_refs:
            max_refs = max(min_refs, max_refs - (max_refs - min_refs))  # 한 번에 2로
            continue

        # 2) 그래도 초과면 ref 길이를 줄인다
        if per_ref_tokens > 192:
            per_ref_tokens = 192
            continue

        # 3) 전체 컨텍스트 한도를 줄인다
        if total_ctx_tokens > 768:
            total_ctx_tokens = 768
            continue

        # 4) 히스토리를 더 자른다
        if previous_messages:
            if len(previous_messages) >= 2:
                previous_messages = previous_messages[2:]
            else:
                previous_messages = []
            continue

        # 5) 생성 길이를 줄인다
        if GEN_TOKENS > 200:
            GEN_TOKENS = 200
            continue

        # 더 줄일 수 없으면 종료(최소 상태)
        break

    # LLM 호출
    llm = ChatOpenAI(
        model_name="unsloth/gemma-3-4b-it",
        openai_api_base="http://vllm.api:8000/v1",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "sk-fake-key"),
    )

    print(f"DEBUG: 최종 토큰 수(메시지): {count_tokens(all_messages)} / GEN={GEN_TOKENS}")

    runtime_params = {
        "max_tokens": GEN_TOKENS,
        "temperature": 0.7,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.05,
        "stop": ["<end_of_turn>"],
    }

    response = llm.invoke(all_messages, **runtime_params)

    # 응답 정리
    cleaned_answer = response.content.split("<end_of_turn>")[0].strip()
    if not cleaned_answer.endswith("."):
        sentences = cleaned_answer.split(".")
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            cleaned_answer = ".".join(sentences[:-1]) + "."

    # 스몰토크는 위에서 return되어 여기 도달하지 않음 → 메모리 오염 방지
    memory.save_context({"input": query}, {"output": cleaned_answer})

    token_usage = response.response_metadata.get("token_usage", {})

    # === (추가) 답변 본문에 출처 섹션 자동 부착 ===
    if references:
        # 문서명 중복 제거 + 상위 3개만
        seen = set()
        top_refs = []
        for r in references:
            doc = r.get("document") or ""
            if doc and doc not in seen:
                top_refs.append((doc, r.get("score", 0.0)))
                seen.add(doc)
            if len(top_refs) >= 3:
                break

        cite_lines = [f"- {doc}" for doc, score in top_refs]
        citation_block = "\n\n[출처]\n" + "\n".join(cite_lines)

        # 너무 길어지지 않게 약간 여유만 둠
        if len(cleaned_answer) + len(citation_block) <= 400:
            cleaned_answer = cleaned_answer.rstrip() + citation_block

    return {
        "answer": cleaned_answer,
        "input_tokens": input_query,
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "references": references[0]["document"] if references else "",
        "rank": references,
    }
