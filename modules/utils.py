import os
import re
from datetime import datetime
from pathlib import Path

from modules import shared
from modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) is list:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    with open(abs_path, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved {abs_path}.')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    if abs_path.exists():
        abs_path.unlink()
        logger.info(f'Deleted {fname}.')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=natural_keys)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml'))], key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompts = []
    files = set((k.stem for k in Path('prompts').glob('*.txt')))
    prompts += sorted([k for k in files if re.match('^[0-9]', k)], key=natural_keys, reverse=True)
    prompts += sorted([k for k in files if re.match('^[^0-9]', k)], key=natural_keys)
    prompts += ['Instruct-' + k for k in get_available_instruction_templates() if k != 'None']
    prompts += ['None']
    return prompts


def get_available_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return ['None'] + sorted(set((k.stem for k in paths if k.stem != "instruction-following")), key=natural_keys)


def get_available_instruction_templates():
    path = "characters/instruction-following"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    return sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)


def get_available_loras():
    return sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)


def get_available_sessions():
    items = sorted(set(k.stem for k in Path('logs').glob(f'session_{shared.get_mode()}*')), key=natural_keys, reverse=True)
    return [item for item in items if 'autosave' in item] + [item for item in items if 'autosave' not in item]
