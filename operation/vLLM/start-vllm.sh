#!/bin/bash

# 필요한 환경 변수 설정
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=miniostorage

# 사용할 모델 이름
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# vLLM 실행
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --chat-template chatml \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --enable-usage-stats \
  --disable-log-stats
