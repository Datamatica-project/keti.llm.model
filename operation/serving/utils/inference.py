from .reranker import load_reranker, rerank_with_bge
from .search import vector_search
from dto.routings import RoutingResult, RouteType

from .buffer import save_session_memory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer

from typing import Dict, Any, List


reranker = load_reranker()

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it", trust_remote_code=True)


def count_tokens(messages: List) -> int:
    """메시지 리스트의 토큰 수 계산"""
    total_tokens = 0
    for message in messages:
        if hasattr(message, 'content'):
            total_tokens += len(tokenizer.encode(message.content))
        else:
            total_tokens += len(tokenizer.encode(str(message)))
    return total_tokens


def route_query(query: str) -> RoutingResult:
    """질문 라우팅 판단"""

    llm = ChatOpenAI(
        model_name="unsloth/gemma-3-4b-it",
        openai_api_base="http://localhost:8000/v1",
        max_tokens=50,
        temperature=0,
        openai_api_key="sk-fake-key"
    )

    prompt = f"""이 질문이 농업 전문 지식/문서 검색이 필요한지 판단하세요.

질문: "{query}"

농업 기술, 작물 재배, 병해충, 농약 등 전문 정보가 필요하면 "농업검색"
일반 대화, 인사, 감사 등은 "일반대화"

답변: """

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if "농업검색" in content:
            route: RouteType = "document_search"
            reasoning = "농업 전문 지식 필요"
        else:
            route = "general_chat"
            reasoning = "일반 대화"

        return RoutingResult(route=route, reasoning=reasoning)

    except Exception as e:
        # 오류시 안전한 기본값
        return RoutingResult(route="document_search", reasoning=f"라우팅 오류: {str(e)}")


def generate_response(query: str, session_id: str) -> Dict[str, Any]:
    """동적 라우팅이 적용된 응답 생성"""

    # Redis 기반 버퍼 메모리 생성

    memory = save_session_memory(session_id, "redis://192.168.0.150:6379")
    if memory is None:
        return {"error": "메모리 초기화 실패"}

    input_query = len(tokenizer.encode(query))


    routing: RoutingResult = route_query(query)
    print(f"라우팅: {routing['route']} - {routing['reasoning']}")

    # 라우팅 결과에 따른 처리
    if routing["route"] == "document_search":
        print(f"농업 검색 수행: {query}")
        vector_results = vector_search(query, top_k=8)
        reranked = rerank_with_bge(query, vector_results, reranker, top_k=5)

        references = [
            {
                "document": doc.get("document"),
                "text": doc.get("text"),
                "score": score
            }
            for doc, score in reranked if score > 0.5
        ]
    else:
        print(f"일반 대화: {query}")
        references = []

    # 프롬프트 생성
    if not references:
        prompt = f"""사용자의 질문에 대해 자세히 설명하세요.:

[질문]
{query}
"""
    else:
        context_docs = references
        context = "\n".join([ref["text"] for ref in context_docs])
        prompt = f"""아래 문서를 참고해서 사용자의 질문에 대해 자세히 설명하세요.:

[문서 요약]
{context}

[질문]
{query}
"""

    # 응답 생성용 LLM (라우팅용과 다른 설정)
    llm = ChatOpenAI(
        model_name="unsloth/gemma-3-4b-it",
        openai_api_base="http://localhost:8000/v1",
        max_tokens=1024,
        temperature=0.1,
        openai_api_key="sk-fake-key"
    )

    # 이전 대화 메시지 불러오기
    previous_messages = memory.load_memory_variables({}).get("chat_history", [])

    # 새 메시지 추가
    current_messages = [
        HumanMessage(content="당신은 한국어로만 답변하는 전문 농업 상담가입니다. 절대로 외국어를 섞지 말고, 반드시 한국어로만 답변하세요.\n\n" + prompt)
    ]

    all_messages = previous_messages + current_messages

    print(f"DEBUG: 초기 토큰 수: {count_tokens(all_messages)}")
    print(f"DEBUG: previous_messages 길이: {len(previous_messages)}")

    max_tokens = 2500

    while count_tokens(all_messages) > max_tokens and len(previous_messages) > 0:
        # 기존
        previous_messages.pop(0)

        # 수정: 순서가 깨지지 않게 2개씩 삭제 (user+assistant 쌍으로)
        if len(previous_messages) >= 2:
            previous_messages.pop(0)  # user 삭제
            previous_messages.pop(0)  # assistant 삭제
        else:
            previous_messages.clear()  # 남은 게 1개면 전체 삭제

        all_messages = previous_messages + current_messages
        print(f"🗑대화 쌍 삭제, 현재 토큰: {count_tokens(all_messages)}")

    print(f"DEBUG: 최종 토큰 수: {count_tokens(all_messages)}")

    response = llm.invoke(all_messages)

    # 응답을 메모리에 추가

    memory.save_context(
        {"input": query},
        {"output": response.content}
    )

    token_usage = response.response_metadata.get("token_usage", {})

    return {
        "answer": response.content,
        "input_tokens": input_query,
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "references": references[0]["document"] if references else "",
        "rank": references
    }