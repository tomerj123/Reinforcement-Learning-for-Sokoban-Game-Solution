import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA device is not available")
