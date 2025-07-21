from fastapi import APIRouter
from dto.inference_schemas import QueryRequest, QueryResponse
from utils.inference import generate_response
from utils.buffer import delete_session_memory

router = APIRouter(
    prefix="/v1",
    tags=["serving"],
)

@router.post("/chat/completions", response_model=QueryResponse)
async def consult_agriculture(request: QueryRequest):
    result = generate_response(request.query, session_id=request.session_id)
    return QueryResponse(**result)

@router.delete("/chat/memory/{session_id}")
async def clear_memory(session_id: str):
    success = delete_session_memory(session_id, "redis://192.168.0.150:6379")
    return {"success": success}
