'''
Converts a transformers model to .pt, which is faster to load.
 
Example:
python convert-to-torch.py models/opt-1.3b
 
The output will be written to torch-dumps/name-of-the-model.pt
'''
 
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration
import torch
from sys import argv
from pathlib import Path
 
path = Path(argv[1])
model_name = path.name

print(f"Loading {model_name}...")
if model_name in ['flan-t5', 't5-large']:
    model = T5ForConditionalGeneration.from_pretrained(path).cuda()
else:
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
print("Model loaded.")

print(f"Saving to torch-dumps/{model_name}.pt")
torch.save(model, Path(f"torch-dumps/{model_name}.pt"))
