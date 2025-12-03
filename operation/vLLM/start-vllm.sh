#!/bin/bash

# 🔧 Gemma 파인튜닝 모델을 위한 vLLM 설정

# 필요한 환경 변수 설정
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=miniostorage
export TOKENIZERS_PARALLELISM=false

# ✅ 파인튜닝된 체크포인트 디렉토리 (컨테이너 안 경로)
MODEL_NAME="/models/checkpoint-2352"

echo "🚀 Gemma-3-4B (finetuned) vLLM 서버 시작 중..."
echo "모델 경로: $MODEL_NAME"

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
  --served-model-name "gemma-3-4b-it-finetuned" \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --skip-mm-profiling
