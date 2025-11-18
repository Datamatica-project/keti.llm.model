from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Reference(BaseModel):
    document: str
    text: str

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    input_tokens: int
    completion_tokens: Optional[int] = None
    references: Optional[str] = None
    rank: List[Dict[str, Any]]

