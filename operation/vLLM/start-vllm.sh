#!/bin/bash

# 🔧 Gemma 모델을 위한 최적화된 vLLM 설정

# 필요한 환경 변수 설정
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=miniostorage
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="unsloth/gemma-3-4b-it"

echo "🚀 Gemma-3-4B vLLM 서버 시작 중..."
echo "모델: $MODEL_NAME"

# vLLM 실행 - Gemma EOS 토큰 처리 최적화
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --enable-force-include-usage \
  --disable-log-stats \
  --trust-remote-code \
  --tokenizer ${MODEL_NAME} \
  --served-model-name "unsloth/gemma-3-4b-it" \
  --gpu-memory-utilization 0.8 \
  --enforce-eager
