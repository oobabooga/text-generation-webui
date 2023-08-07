import re
from pathlib import Path

import yaml

from modules import utils
from modules.text_generation import get_encoded_length


def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    elif fname.startswith('Instruct-'):
        fname = re.sub('^Instruct-', '', fname)
        file_path = Path(f'instruction-templates/{fname}.yaml')
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            output = ''
            if 'context' in data:
                output += data['context']

            replacements = {
                '<|user|>': data['user'],
                '<|bot|>': data['bot'],
                '<|user-message|>': 'Input',
            }

            output += utils.replace_all(data['turn_template'].split('<|bot-message|>')[0], replacements)
            return output.rstrip(' ')
    else:
        file_path = Path(f'prompts/{fname}.txt')
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]

            return text


def count_tokens(text):
    try:
        tokens = get_encoded_length(text)
        return f'{tokens} tokens in the input.'
    except:
        return 'Couldn\'t count the number of tokens. Is a tokenizer loaded?'
