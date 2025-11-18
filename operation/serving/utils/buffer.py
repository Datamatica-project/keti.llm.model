from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

def delete_session_memory(session_id: str, redis_url:str):
    try:
        memory = RedisChatMessageHistory(session_id=session_id, url=redis_url)
        memory.clear()
        return True
    except Exception as e:
        print(f"메모리 처리 실패 : {e}")
        return False

def save_session_memory(session_id: str, redis_url: str) -> ConversationBufferMemory:
    try:
        chat_history = RedisChatMessageHistory(session_id=session_id, url=redis_url)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=chat_history,
            return_messages=True
        )
        return memory
    except Exception as e:
        print(f"메모리 저장 실패 : {e}")
        return None