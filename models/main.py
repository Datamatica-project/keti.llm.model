import torch

print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("사용 가능한 GPU 수:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("현재 사용 중인 GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA를 사용할 수 없습니다.")