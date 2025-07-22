FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install transformers datasets jupyterlab ipywidgets unsloth accelerate bitsandbytes

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
