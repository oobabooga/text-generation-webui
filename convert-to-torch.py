'''
Converts a transformers model to .pt, which is faster to load.
 
Run with python convert.py /path/to/model/
Make sure to write /path/to/model/ with a trailing / and not
/path/to/model
 
Output will be written to torch-dumps/name-of-the-model.pt
'''
 
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, OPTForCausalLM, AutoTokenizer, set_seed
from transformers import GPT2Tokenizer, GPT2Model, T5Tokenizer, T5ForConditionalGeneration
import torch
import sys
from sys import argv
import time
import glob
import psutil
 
print(f"torch-dumps/{argv[1].split('/')[-2]}.pt")
 
if argv[1].endswith('pt'):
    model = OPTForCausalLM.from_pretrained(argv[1], device_map="auto")
    torch.save(model, f"torch-dumps/{argv[1].split('/')[-2]}.pt")
elif 'galactica' in argv[1].lower():
    model = OPTForCausalLM.from_pretrained(argv[1], low_cpu_mem_usage=True, torch_dtype=torch.float16)
    #model = OPTForCausalLM.from_pretrained(argv[1], low_cpu_mem_usage=True, load_in_8bit=True)
    torch.save(model, f"torch-dumps/{argv[1].split('/')[-2]}.pt")
elif 'flan-t5' in argv[1].lower():
    model = T5ForConditionalGeneration.from_pretrained(argv[1], low_cpu_mem_usage=True, torch_dtype=torch.float16)
    torch.save(model, f"torch-dumps/{argv[1].split('/')[-2]}.pt")
else:
    print("Loading the model")
    model = AutoModelForCausalLM.from_pretrained(argv[1], low_cpu_mem_usage=True, torch_dtype=torch.float16)
    print("Model loaded")
    #model = AutoModelForCausalLM.from_pretrained(argv[1], device_map='auto', load_in_8bit=True)
    torch.save(model, f"torch-dumps/{argv[1].split('/')[-2]}.pt")

