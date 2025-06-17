from pathlib import Path

from modules import shared, utils
from modules.text_generation import get_encoded_length


def load_prompt(fname):
    if not fname:
        # Create new file
        new_name = utils.current_time()
        prompt_path = Path("user_data/logs/notebook") / f"{new_name}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        initial_content = "In this story,"
        prompt_path.write_text(initial_content, encoding='utf-8')

        # Update settings to point to new file
        shared.settings['prompt-notebook'] = new_name

        return initial_content

    file_path = Path(f'user_data/logs/notebook/{fname}.txt')
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if len(text) > 0 and text[-1] == '\n':
                text = text[:-1]

            return text
    else:
        return ''


def count_tokens(text):
    try:
        tokens = get_encoded_length(text)
        return str(tokens)
    except:
        return '0'
