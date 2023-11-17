from pathlib import Path

from modules.text_generation import get_encoded_length


def load_prompt(fname):
    if fname in ['None', '']:
        return ''
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
        return str(tokens)
    except:
        return '0'
