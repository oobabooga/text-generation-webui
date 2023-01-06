import os
import re
import time
import glob
from sys import exit
import torch
import argparse
import gradio as gr
import transformers
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, OPTForCausalLM, T5Tokenizer, T5ForConditionalGeneration, GPTJModel, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Name of the model to load by default')
args = parser.parse_args()
loaded_preset = None
available_models = sorted(set(map(lambda x : x.split('/')[-1].replace('.pt', ''), glob.glob("models/*[!\.][!t][!x][!t]")+ glob.glob("torch-dumps/*[!\.][!t][!x][!t]"))))

def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    # Loading the model
    if os.path.exists(f"torch-dumps/{model_name}.pt"):
        print("Loading in .pt format...")
        model = torch.load(f"torch-dumps/{model_name}.pt").cuda()
    elif model_name.lower().startswith(('gpt-neo', 'opt-', 'galactica')):
        if any(size in model_name for size in ('13b', '20b', '30b')):
            model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", device_map='auto', load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
    elif model_name in ['gpt-j-6B']:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
    elif model_name in ['flan-t5', 't5-large']:
        model = T5ForConditionalGeneration.from_pretrained(f"models/{model_name}").cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()

    # Loading the tokenizer
    if model_name.startswith('gpt4chan'):
        tokenizer = AutoTokenizer.from_pretrained("models/gpt-j-6B/")
    elif model_name in ['flan-t5']:
        tokenizer = T5Tokenizer.from_pretrained(f"models/{model_name}/")
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}/")

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

# Removes empty replies from gpt4chan outputs
def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)
    return s

def generate_reply(question, temperature, max_length, inference_settings, selected_model):
    global model, tokenizer, model_name, loaded_preset, preset

    if selected_model != model_name:
        model_name = selected_model
        model = None
        tokenier = None
        torch.cuda.empty_cache()
        model, tokenizer = load_model(model_name)
    if inference_settings != loaded_preset:
        with open(f'presets/{inference_settings}.txt', 'r') as infile:
            preset = infile.read()
        loaded_preset = inference_settings

    torch.cuda.empty_cache()
    input_text = question
    input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

    output = eval(f"model.generate(input_ids, {preset}).cuda()")

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    if model_name.startswith('gpt4chan'):
        reply = fix_gpt4chan(reply)

    return reply

# Choosing the default model
if args.model is not None:
    model_name = args.model
else:
    if len(available_models == 0):
        print("No models are available! Please download at least one.")
        exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print("The following models are available:\n")
        for i,model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input())-1
    model_name = available_models[i]
model, tokenizer = load_model(model_name)

if model_name.startswith('gpt4chan'):
    default_text = "-----\n--- 865467536\nInput text\n--- 865467537\n"
else:
    default_text = "Common sense questions and answers\n\nQuestion: \nFactual answer:"

interface = gr.Interface(
    generate_reply,
    inputs=[
        gr.Textbox(value=default_text, lines=15),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Temperature', value=0.7),
        gr.Slider(minimum=1, maximum=2000, step=1, label='max_length', value=200),
        gr.Dropdown(choices=list(map(lambda x : x.split('/')[-1].split('.')[0], glob.glob("presets/*.txt"))), value="Default"),
        gr.Dropdown(choices=available_models, value=model_name),
    ],
    outputs=[
         gr.Textbox(placeholder="", lines=15),
    ],
    title="Text generation lab",
    description=f"Generate text using Large Language Models.",
)

interface.launch(share=False, server_name="0.0.0.0")
