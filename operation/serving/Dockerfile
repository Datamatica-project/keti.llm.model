# CUDA 런타임 포함된 PyTorch 이미지 사용
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    gcc g++ pkg-config build-essential curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Seoul
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
