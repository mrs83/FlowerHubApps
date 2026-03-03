import torch
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version (torch): {torch.version.hip}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

print("Attempting to load model with 4-bit quantization...")
model_id = "microsoft/Phi-4-mini-instruct"

# Using BitsAndBytesConfig as provided in standard examples
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

try:
    # Testing without trust_remote_code first to see if native transformers 5.2.0 works
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=False 
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
