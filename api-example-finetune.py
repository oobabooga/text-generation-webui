import json
import requests

HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/finetune'

def finetune(lora_name, raw_text_file):
    request = {
        'lora_name': lora_name,
        'raw_text_file': raw_text_file,
    }

    response = requests.post(URI, json=request)

if __name__ == '__main__':
    lora_name = 'lora'
    raw_text_file = 'input'
    finetune(lora_name, raw_text_file)