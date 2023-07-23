import json

import requests

HOST = 'localhost:5000'
URI_CHAT_TOKEN_COUNT = f'http://{HOST}/api/v1/chat-token-count'
URI_CHAT = f'http://{HOST}/api/v1/chat'
MAX_SEQ_LEN = 2048 # Only for demonstration. You can fetch
                    # the actual max_seq_len value of the running
                    # model by calling /api/v1/model 

def api_chat(user_input, history, max_new_tokens):
    request = {
        'user_input': user_input,
        'max_new_tokens': max_new_tokens,
        'history': history,
        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': 'Example',
        # 'context_instruct': '',  # Optional
        'your_name': 'You',

        'regenerate': False,
        '_continue': False,
        'stop_at_newline': False,
        'chat_generation_attempts': 1,
        'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': MAX_SEQ_LEN,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI_CHAT, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
        return result['visible'][-1][1]
    raise Exception("Error while calling chat API")

def api_chat_count_tokens(user_input, history):
    request = {
        "user_input": user_input,
        "history": history,
        "mode": "instruct",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "_continue": False,
    }

    response = requests.post(URI_CHAT_TOKEN_COUNT, json=request)
    if response.status_code == 200:
        return response.json()['results'][0]['tokens']
    raise Exception("Error while calling chat-count-tokens API")

if __name__ == '__main__':

    user_input = "Please give me a step-by-step guide on how to plant a tree in my backyard."
    history = {'internal': [], 'visible': []}

    num_prompt_tokens = api_chat_count_tokens(user_input, history)
    print(f"Number of tokens in prompt: {num_prompt_tokens}")

    # In order to estimate a valid max_new_tokens value we need to know
    # how many tokens the prompt consumes (using the chat-token-count API endpoint)
    # and the model's max_seq_len. This way we can take advantage of the
    # full context without running the risk of unknowingly truncating part of the prompt.
    max_new_tokens = MAX_SEQ_LEN - num_prompt_tokens

    print(f"Number of max_new_tokens: {max_new_tokens}")
    answer = api_chat(user_input, history, max_new_tokens)
    print(f"\nAnswer: {answer}")
