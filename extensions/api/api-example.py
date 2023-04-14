'''

This is an example on how to use the API extension.

Make sure to start the web UI with the following flags:

python server.py --model MODEL --listen --no-stream --extensions api

'''
import json

import requests

# Server address
server = "127.0.0.1"
port = 5000

# Input prompt
prompt = "What I would like to say is the following: "

# Generation parameters
# Reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
params = {
    'max_new_tokens': 200,
    'do_sample': True,
    'temperature': 0.72,
    'top_p': 0.73,
    'typical_p': 1,
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

response = requests.post(f"http://{server}:{port}/api/v1/generate", json={
    "data": {
        "prompt": json.dumps(prompt),
        "params": json.dumps(params)
    }
}).json()

reply = response["data"][0]
print(reply)
