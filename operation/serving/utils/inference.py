from .reranker import load_reranker, rerank_with_bge
from .search import vector_search

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage

from transformers import AutoTokenizer

from typing import Dict, Any

reranker = load_reranker()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

def generate_response(query: str, session_id: str) -> Dict[str, Any]:
    # Redis 기반 버퍼 메모리 생성
    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://192.168.0.150:6379"
    )
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True
    )

    input_query = len(tokenizer.encode(query))

    vector_results = vector_search(query, top_k=15)
    reranked = rerank_with_bge(query, vector_results, reranker, top_k=5)

    references = [
        {
            "document": doc.get("document"),
            "text": doc.get("text"),
            "score": score
        }
        for doc, score in reranked if score > 0.5
    ]

    if not references:
        prompt = f"""사용자의 질문에 대해 반드시 한국어로 자세히 설명하세요 절대 외국어를 섞으면 안됩니다.:

[질문]
{query}
"""
    else:
        context_docs = references[:3]
        context = "\n".join([ref["text"] for ref in context_docs])
        prompt = f"""아래 문서를 참고해서 사용자의 질문에 반드시 한국어로 자세히 답하세요 절대 외국어를 섞으면 안됩니다.:

[문서 요약]
{context}

[질문]
{query}
"""

    llm = ChatOpenAI(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        openai_api_base="http://localhost:8000/v1",
        max_tokens=2048,
        openai_api_key="sk-fake-key"
    )

    # 이전 대화 메시지 불러오기
    previous_messages = memory.load_memory_variables({}).get("history", [])

    # 새 메시지 추가
    current_messages = [
        SystemMessage(content="당신은 한국어로만 답변하는 전문 농업 상담가입니다. 절대로 외국어를 섞지 말고, 반드시 한국어로만 답변하세요."),
        HumanMessage(content=prompt)
    ]

    full_messages = previous_messages + current_messages

    # 응답 생성
    response = llm.invoke(full_messages)

    # 응답을 메모리에 추가
    memory.save_context(
        {"input": query},
        {"output": response.content}
    )

    token_usage = response.response_metadata["token_usage"]

    return {
        "answer": response.content,
        "input_tokens": input_query,
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "references": references[0]["document"] if references else "",
        "rank": references
    }
