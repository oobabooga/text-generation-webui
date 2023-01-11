'''
Converts a transformers model to .pt, which is faster to load.
 
Example:
python convert-to-torch.py models/opt-1.3b
 
The output will be written to torch-dumps/name-of-the-model.pt
'''
 
from transformers import AutoModelForCausalLM
import torch
from sys import argv
from pathlib import Path
 
path = Path(argv[1])
model_name = path.name

print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
print("Model loaded.")

print(f"Saving to torch-dumps/{model_name}.pt")
torch.save(model, Path(f"torch-dumps/{model_name}.pt"))
