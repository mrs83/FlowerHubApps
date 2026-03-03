import ray
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
ray.init()
print("Ray Resources:", ray.available_resources())
ray.shutdown()
