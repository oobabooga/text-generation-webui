'''

This is an example on how to use the API for oobabooga/text-generation-webui.

Make sure to start the web UI with the following flags:

python server.py --model MODEL --listen --no-stream --extensions api

Optionally, you can also add the --share flag to generate a public gradio URL,
allowing you to use the API remotely.

'''
import json

import requests

from deep_translator import GoogleTranslator, LibreTranslator

# Server address
server = "127.0.0.1"

# Generation parameters
# Reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
params = {
    'max_new_tokens': 100,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.7,
    'typical_p': 0.2,
    'repetition_penalty': 1.1,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 1,
    'early_stopping': False,
    'seed': -1,
    'add_bos_token': True,
    'custom_stopping_strings': [],
    'truncation_length': 2048,
    'ban_eos_token': False,
}

from time import time
start = time()
# Input promptw for Alpaca
tpl = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
tpl += "### Instruction:\n{0}\n"
tpl += "### Response:"
prompt = "Who is Barack Obama?"
#prompt = GoogleTranslator(source="es", target='en').translate(prompt)

payload = json.dumps([prompt, params])

response = requests.post(f"http://{server}:5000/api/v1/generate", json={
    "prompt": tpl.format(prompt),
    'max_new_tokens': 100,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.95,
    'typical_p': 1,
    'repetition_penalty': 1.3,
}).json()

print(response)
reply = response["results"][0]['text']

#reply = GoogleTranslator(source="en", target='es').translate(reply)
print(reply)
end = time()
print("Duration: {0}".format(end-start))