import os
import re
import time
import glob
import torch
import gradio as gr
import transformers
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, OPTForCausalLM, T5Tokenizer, T5ForConditionalGeneration, GPTJModel, AutoModel

#model_name = "bloomz-7b1-p3"
#model_name = 'gpt-j-6B-float16'
#model_name = "opt-6.7b"
#model_name = 'opt-13b'
#model_name = "gpt4chan_model_float16"
model_name = 'galactica-6.7b'
#model_name = 'gpt-neox-20b'
#model_name = 'flan-t5'
#model_name = 'OPT-13B-Erebus'

loaded_preset = None

def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    if os.path.exists(f"torch-dumps/{model_name}.pt"):
        print("Loading in .pt format...")
        model = torch.load(f"torch-dumps/{model_name}.pt").cuda()
    elif model_name in ['gpt-neox-20b', 'opt-13b', 'OPT-13B-Erebus']:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", device_map='auto', load_in_8bit=True)
    elif model_name in ['gpt-j-6B']:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
    elif model_name in ['flan-t5', 't5-large']:
        model = T5ForConditionalGeneration.from_pretrained(f"models/{model_name}").cuda()

    if model_name in ['gpt4chan_model_float16']:
        tokenizer = AutoTokenizer.from_pretrained("models/gpt-j-6B/")
    elif model_name in ['flan-t5']:
        tokenizer = T5Tokenizer.from_pretrained(f"models/{model_name}/")
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}/")

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)

    return s

def fn(question, temperature, max_length, inference_settings, selected_model):
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

model, tokenizer = load_model(model_name)
if model_name.startswith('gpt4chan'):
    default_text = "-----\n--- 865467536\nInput text\n--- 865467537\n"
else:
    default_text = "Common sense questions and answers\n\nQuestion: \nFactual answer:"

interface = gr.Interface(
    fn,
    inputs=[
        gr.Textbox(value=default_text, lines=15),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Temperature', value=0.7),
        gr.Slider(minimum=1, maximum=2000, step=1, label='max_length', value=200),
        gr.Dropdown(choices=list(map(lambda x : x.split('/')[-1].split('.')[0], glob.glob("presets/*.txt"))), value="Default"),
        gr.Dropdown(choices=sorted(set(map(lambda x : x.split('/')[-1].replace('.pt', ''), glob.glob("models/*") + glob.glob("torch-dumps/*")))), value=model_name),
    ],
    outputs=[
         gr.Textbox(placeholder="", lines=15),
    ],
    title="Text generation lab",
    description=f"Generate text using Large Language Models. Currently working with {model_name}",
)

interface.launch(share=False, server_name="0.0.0.0")
