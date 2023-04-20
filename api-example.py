'''
This is an example on how to use the API for oobabooga/text-generation-webui.
Make sure to start the web UI with the following flags:
python server.py --model MODEL --extensions api
Optionally, you can also add the --share flag to generate a public gradio URL,
allowing you to use the API remotely.
'''
import requests



# Server address
server = "127.0.0.1"
port = 5000
prompt = "What I would like to say is the following: "

# Generation parameters
# Reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
params = {
    'prompt': prompt,
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
    'truncation_length': 2048,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'stopping_strings': [],
}


response = requests.post(f"http://{server}:{port}/api/v1/generate", json=params).json()
reply = response['results'][0]['text']
print(reply)
