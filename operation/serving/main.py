import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.inference_router import router
app = FastAPI(
    title="농업 지식 상담 API",
    version="1.0.0",
    description="Vector Search + Reranker + Qwen 기반 농업 상담 서비스",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

