import os
import re
from datetime import datetime
from pathlib import Path

from modules import github, shared
from modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    with open(abs_path_str, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved \"{abs_path_str}\".')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path_str = os.path.abspath(fname)
    rel_path_str = os.path.relpath(abs_path_str, root_folder)
    rel_path = Path(rel_path_str)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    if rel_path.exists():
        rel_path.unlink()
        logger.info(f'Deleted \"{fname}\".')


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
    # Get all GGUF files
    gguf_files = get_available_ggufs()

    # Filter out non-first parts of multipart GGUF files
    filtered_gguf_files = []
    for gguf_path in gguf_files:
        filename = os.path.basename(gguf_path)

        match = re.search(r'-(\d+)-of-\d+\.gguf$', filename)

        if match:
            part_number = match.group(1)
            # Keep only if it's part 1
            if part_number.lstrip("0") == "1":
                filtered_gguf_files.append(gguf_path)
        else:
            # Not a multi-part file
            filtered_gguf_files.append(gguf_path)

    model_dir = Path(shared.args.model_dir)

    # Find top-level directories containing GGUF files
    dirs_with_gguf = set()
    for gguf_path in gguf_files:
        path = Path(gguf_path)
        if len(path.parts) > 0:
            dirs_with_gguf.add(path.parts[0])

    # Find directories with safetensors files
    dirs_with_safetensors = set()
    for item in os.listdir(model_dir):
        item_path = model_dir / item
        if item_path.is_dir():
            if any(file.lower().endswith(('.safetensors', '.pt')) for file in os.listdir(item_path) if (item_path / file).is_file()):
                dirs_with_safetensors.add(item)

    # Find valid model directories
    model_dirs = []
    for item in os.listdir(model_dir):
        item_path = model_dir / item
        if not item_path.is_dir():
            continue

        # Include directory if it either doesn't contain GGUF files
        # or contains both GGUF and safetensors files
        if item not in dirs_with_gguf or item in dirs_with_safetensors:
            model_dirs.append(item)

    model_dirs = sorted(model_dirs, key=natural_keys)

    return ['None'] + filtered_gguf_files + model_dirs


def get_available_ggufs():
    model_list = []
    model_dir = Path(shared.args.model_dir)

    for dirpath, _, files in os.walk(model_dir, followlinks=True):
        for file in files:
            if file.lower().endswith(".gguf"):
                model_path = Path(dirpath) / file
                rel_path = model_path.relative_to(model_dir)
                model_list.append(str(rel_path))

    return sorted(model_list, key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('user_data/presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompt_files = list(Path('user_data/prompts').glob('*.txt'))
    sorted_files = sorted(prompt_files, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [file.stem for file in sorted_files]
    prompts.append('None')
    return prompts


def get_available_characters():
    paths = (x for x in Path('user_data/characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_instruction_templates():
    path = "user_data/instruction-templates"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    extensions = sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)
    extensions = [v for v in extensions if v not in github.new_extensions]
    return extensions


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    # include subdirectories for raw txt files to allow training from a subdirectory of txt files
    if ext == "txt":
        return ['None'] + sorted(set([k.stem for k in list(Path(path).glob('*.txt')) + list(Path(path).glob('*/')) if k.stem != 'put-trainer-datasets-here']), key=natural_keys)

    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)


def get_available_grammars():
    return ['None'] + sorted([item.name for item in list(Path('user_data/grammars').glob('*.gbnf'))], key=natural_keys)
