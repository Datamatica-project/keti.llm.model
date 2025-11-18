FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

WORKDIR /app

# 런타임 유틸 (git: unsloth 설치용, curl: 확인/디버깅용)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 개행 정리 + 실행권한
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 기본 청킹 파라미터(환경변수로 덮어쓰기 가능)
ENV IN_DIR=/app/org_data \
    HTML_DIR=/app/html_data \
    JSON_DIR=/app/data \
    OUT_DIR=/app/out \
    PDF_JOBS=4 \
    CHUNK_MODE=length \
    MAX_TOKENS=700 \
    OVERLAP=100 \
    TOKENIZER_NAME=google/gemma-2-2b \
    MODEL_NAME=unsloth/gemma-3-4b-it \
    MIN_CHARS=50 \
    CAP_TOKENS=800

# 원큐 파이프라인 실행
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
