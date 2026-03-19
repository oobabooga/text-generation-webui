import os
import re
from datetime import datetime
from pathlib import Path

from modules import shared
from modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def sanitize_filename(name):
    """Strip path traversal components from a filename.

    Returns only the final path component with leading dots removed,
    preventing directory traversal via '../' or absolute paths.
    """
    name = Path(name).name  # drop all directory components
    name = name.lstrip('.')  # remove leading dots
    return name


def _is_path_allowed(abs_path_str):
    """Check if a path is under the configured user_data directory."""
    abs_path = Path(abs_path_str).resolve()
    user_data_resolved = shared.user_data_dir.resolve()
    try:
        abs_path.relative_to(user_data_resolved)
        return True
    except ValueError:
        return False


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    abs_path_str = os.path.abspath(fname)
    if not _is_path_allowed(abs_path_str):
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    if Path(abs_path_str).suffix.lower() not in ('.yaml', '.yml', '.json', '.txt', '.gbnf'):
        logger.error(f'Refusing to save file with disallowed extension: \"{fname}\"')
        return

    with open(abs_path_str, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved \"{abs_path_str}\".')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    abs_path_str = os.path.abspath(fname)
    if not _is_path_allowed(abs_path_str):
        logger.error(f'Invalid file path: \"{fname}\"')
        return

    p = Path(abs_path_str)
    if p.exists():
        p.unlink()
        logger.info(f'Deleted \"{fname}\".')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def check_model_loaded():
    if shared.model_name == 'None' or shared.model is None:
        if len(get_available_models()) == 0:
            error_msg = f"No model is loaded.\n\nTo get started:\n1) Place a GGUF file in your {shared.user_data_dir}/models folder\n2) Go to the Model tab and select it"
            logger.error(error_msg)
            return False, error_msg
        else:
            error_msg = "No model is loaded. Please select one in the Model tab."
            logger.error(error_msg)
            return False, error_msg

    return True, None


def resolve_model_path(model_name_or_path, image_model=False):
    """
    Resolves a model path, checking for a direct path
    before the default models directory.
    """

    path_candidate = Path(model_name_or_path)
    if path_candidate.exists():
        return path_candidate
    elif image_model:
        return Path(f'{shared.args.image_model_dir}/{model_name_or_path}')
    else:
        return Path(f'{shared.args.model_dir}/{model_name_or_path}')


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

    return filtered_gguf_files + model_dirs


def get_available_image_models():
    model_dir = Path(shared.args.image_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Find valid model directories
    model_dirs = []
    for item in os.listdir(model_dir):
        item_path = model_dir / item
        if not item_path.is_dir():
            continue

        model_dirs.append(item)

    model_dirs = sorted(model_dirs, key=natural_keys)

    return model_dirs


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


def get_available_mmproj():
    mmproj_dir = shared.user_data_dir / 'mmproj'
    if not mmproj_dir.exists():
        return ['None']

    mmproj_files = []
    for item in mmproj_dir.iterdir():
        if item.is_file() and item.suffix.lower() in ('.gguf', '.bin'):
            mmproj_files.append(item.name)

    return ['None'] + sorted(mmproj_files, key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in (shared.user_data_dir / 'presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    notebook_dir = shared.user_data_dir / 'logs' / 'notebook'
    notebook_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = list(notebook_dir.glob('*.txt'))
    if not prompt_files:
        new_name = current_time()
        new_path = notebook_dir / f"{new_name}.txt"
        new_path.write_text("In this story,", encoding='utf-8')
        prompt_files = [new_path]

    sorted_files = sorted(prompt_files, key=lambda x: x.stat().st_mtime, reverse=True)
    prompts = [file.stem for file in sorted_files]
    return prompts


def get_available_characters():
    paths = (x for x in (shared.user_data_dir / 'characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_users():
    users_dir = shared.user_data_dir / 'users'
    users_dir.mkdir(parents=True, exist_ok=True)
    paths = (x for x in users_dir.iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_instruction_templates():
    path = str(shared.user_data_dir / "instruction-templates")
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    # User extensions (higher priority)
    user_extensions = []
    user_ext_path = shared.user_data_dir / 'extensions'
    if user_ext_path.exists():
        user_exts = map(lambda x: x.parent.name, user_ext_path.glob('*/script.py'))
        user_extensions = sorted(set(user_exts), key=natural_keys)

    # System extensions (excluding those overridden by user extensions)
    system_exts = map(lambda x: x.parent.name, Path('extensions').glob('*/script.py'))
    system_extensions = sorted(set(system_exts) - set(user_extensions), key=natural_keys)

    return user_extensions + system_extensions


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    # include subdirectories for raw txt files to allow training from a subdirectory of txt files
    if ext == "txt":
        return ['None'] + sorted(set([k.stem for k in list(Path(path).glob('*.txt')) + list(Path(path).glob('*/')) if k.stem != 'put-trainer-datasets-here']), key=natural_keys)

    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_chat_datasets(path: str):
    """List JSON datasets that contain chat conversations (messages or ShareGPT format)."""
    return ['None'] + sorted(set([k.stem for k in Path(path).glob('*.json') if k.stem != 'put-trainer-datasets-here' and _is_chat_dataset(k)]), key=natural_keys)


def get_text_datasets(path: str):
    """List JSON datasets that contain raw text ({"text": ...} format)."""
    return ['None'] + sorted(set([k.stem for k in Path(path).glob('*.json') if k.stem != 'put-trainer-datasets-here' and _is_text_dataset(k)]), key=natural_keys)


def _peek_json_keys(filepath):
    """Read the first object in a JSON array file and return its keys."""
    import json
    decoder = json.JSONDecoder()
    WS = ' \t\n\r'
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            buf = ''
            obj_start = None
            while len(buf) < 1 << 20:  # Read up to 1MB
                chunk = f.read(8192)
                if not chunk:
                    break
                buf += chunk
                if obj_start is None:
                    idx = 0
                    while idx < len(buf) and buf[idx] in WS:
                        idx += 1
                    if idx >= len(buf):
                        continue
                    if buf[idx] != '[':
                        return set()
                    idx += 1
                    while idx < len(buf) and buf[idx] in WS:
                        idx += 1
                    if idx >= len(buf):
                        continue
                    obj_start = idx
                try:
                    obj, _ = decoder.raw_decode(buf, obj_start)
                    if isinstance(obj, dict):
                        return set(obj.keys())
                    return set()
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return set()


def _is_chat_dataset(filepath):
    keys = _peek_json_keys(filepath)
    return bool(keys & {'messages', 'conversations'})


def _is_text_dataset(filepath):
    keys = _peek_json_keys(filepath)
    return 'text' in keys


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)


def get_available_grammars():
    return ['None'] + sorted([item.name for item in list((shared.user_data_dir / 'grammars').glob('*.gbnf'))], key=natural_keys)
