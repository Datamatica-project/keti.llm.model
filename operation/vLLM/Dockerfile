FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# 기본 의존성 설치
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    python3 python3-pip \
    awscli

# 최신 pip로 업그레이드
RUN pip install --upgrade pip

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-pip ray

RUN pip install --no-cache-dir \
    vllm \
    fastapi \
    pydantic \
    boto3==1.26.8

COPY start-vllm.sh /app/start-vllm.sh

RUN chmod +x /app/start-vllm.sh

EXPOSE 8000

CMD ["/app/start-vllm.sh"]
