import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

model_id = "microsoft/Phi-4-mini-instruct"

try:
    print("Attempting to load model in bfloat16 (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully in bfloat16!")
except Exception as e:
    print(f"Error loading model: {e}")
