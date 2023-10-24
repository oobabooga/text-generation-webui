import asyncio
import html
import json
import sys

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")

# For local streaming, the websockets are hosted without ssl - ws://
HOST = 'localhost:5005'
URI = f'ws://{HOST}/api/v1/chat-stream'

# For reverse-proxied streaming, the remote will likely host with ssl - wss://
# URI = 'wss://your-uri-here.trycloudflare.com/api/v1/stream'


async def run(user_input, history):
    # Note: the selected defaults change from time to time.
    request = {
        'user_input': user_input,
        'max_new_tokens': 250,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,
        'history': history,
        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': 'Example',
        'instruction_template': 'Vicuna-v1.1',  # Will get autodetected if unset
        'your_name': 'You',
        # 'name1': 'name of user', # Optional
        # 'name2': 'name of character', # Optional
        # 'context': 'character context', # Optional
        # 'greeting': 'greeting', # Optional
        # 'name1_instruct': 'You', # Optional
        # 'name2_instruct': 'Assistant', # Optional
        # 'context_instruct': 'context_instruct', # Optional
        # 'turn_template': 'turn_template', # Optional
        'regenerate': False,
        '_continue': False,
        'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

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
        'presence_penalty': 0,
        'frequency_penalty': 0,
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
        'grammar_string': '',
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'custom_token_bans': '',
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['history']
                case 'stream_end':
                    return


async def print_response_stream(user_input, history):
    cur_len = 0
    async for new_history in run(user_input, history):
        cur_message = new_history['visible'][-1][1][cur_len:]
        cur_len += len(cur_message)
        print(html.unescape(cur_message), end='')
        sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.


if __name__ == '__main__':
    user_input = "Please give me a step-by-step guide on how to plant a tree in my backyard."

    # Basic example
    history = {'internal': [], 'visible': []}

    # "Continue" example. Make sure to set '_continue' to True above
    # arr = [user_input, 'Surely, here is']
    # history = {'internal': [arr], 'visible': [arr]}

    asyncio.run(print_response_stream(user_input, history))
