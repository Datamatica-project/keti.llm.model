import torch
print("cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
