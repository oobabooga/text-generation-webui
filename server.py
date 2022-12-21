import time
import re
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

def load_model(model_name):
    print(f"Loading {model_name}")

    t0 = time.time()
    if model_name in ['gpt-neox-20b', 'opt-13b', 'OPT-13B-Erebus']:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", device_map='auto', load_in_8bit=True)
    elif model_name in ['gpt-j-6B']:
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}", low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
    elif model_name in ['flan-t5']:
        model = T5ForConditionalGeneration.from_pretrained(f"models/{model_name}").cuda()
    else:
        model = torch.load(f"torch-dumps/{model_name}.pt").cuda()

    if model_name in ['gpt4chan_model_float16']:
        tokenizer = AutoTokenizer.from_pretrained("models/gpt-j-6B/")
    elif model_name in ['flan-t5']:
        tokenizer = T5Tokenizer.from_pretrained(f"models/{model_name}/")
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}/")

    print(f"Loaded the model in {time.time()-t0} seconds.")
    return model, tokenizer

def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)

    return s

def fn(question, temperature, max_length, inference_settings, selected_model):
    global model, tokenizer, model_name

    if selected_model != model_name:
        model_name = selected_model
        model = None
        tokenier = None
        torch.cuda.empty_cache()
        model, tokenizer = load_model(model_name)

    torch.cuda.empty_cache()
    input_text = question
    input_ids = tokenizer.encode(str(input_text), return_tensors='pt').cuda()

    if inference_settings == 'Default':
        output = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=max_length,
            #max_length=max_length+len(input_ids[0]),
            top_p=1,
            typical_p=0.3,
            temperature=temperature, 
        ).cuda()
    elif inference_settings == 'Verbose':
        output = model.generate(
            input_ids,
            num_beams=10,
            min_length=max_length,
            max_new_tokens=max_length,
            length_penalty =1.4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.7,
            top_k=150,
            top_p=0.92,
            repetition_penalty=4.5,
        ).cuda()

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
        gr.Dropdown(choices=["Default", "Verbose"], value="Default"),
        gr.Dropdown(choices=["gpt4chan_model_float16", "galactica-6.7b", "opt-6.7b",  "opt-13b", "gpt-neox-20b", "gpt-j-6B-float16", "flan-t5", "bloomz-7b1-p3", "OPT-13B-Erebus"], value=model_name),
    ],
    outputs=[
         gr.Textbox(placeholder="", lines=15),
    ],
    title="Text generation lab",
    description=f"Generate text using Large Language Models. Currently working with {model_name}",
)

interface.launch(share=False, server_name="0.0.0.0")
