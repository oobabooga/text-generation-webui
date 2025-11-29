from pathlib import Path

import yaml

import modules.shared as shared
from modules.logging_colors import logger

DEFAULTS = {
    'model_name': 'None',
    'dtype': 'bfloat16',
    'attn_backend': 'sdpa',
    'cpu_offload': False,
    'compile_model': False,
}


def get_settings_path():
    """Get the path to the image model settings file."""
    return Path(shared.args.image_model_dir) / 'settings.yaml'


def load_yaml_settings():
    """Load raw settings from yaml file."""
    settings_path = get_settings_path()

    if not settings_path.exists():
        return {}

    try:
        with open(settings_path, 'r') as f:
            saved = yaml.safe_load(f)
            return saved if saved else {}
    except Exception as e:
        logger.warning(f"Failed to load image model settings: {e}")
        return {}


def get_effective_settings():
    """
    Get effective settings with precedence:
    1. CLI flag (if provided)
    2. Saved yaml value (if exists)
    3. Hardcoded default

    Returns a dict with all settings.
    """
    yaml_settings = load_yaml_settings()

    effective = {}

    # model_name: CLI --image-model > yaml > default
    if shared.args.image_model:
        effective['model_name'] = shared.args.image_model
    else:
        effective['model_name'] = yaml_settings.get('model_name', DEFAULTS['model_name'])

    # dtype: CLI --image-dtype > yaml > default
    if shared.args.image_dtype is not None:
        effective['dtype'] = shared.args.image_dtype
    else:
        effective['dtype'] = yaml_settings.get('dtype', DEFAULTS['dtype'])

    # attn_backend: CLI --image-attn-backend > yaml > default
    if shared.args.image_attn_backend is not None:
        effective['attn_backend'] = shared.args.image_attn_backend
    else:
        effective['attn_backend'] = yaml_settings.get('attn_backend', DEFAULTS['attn_backend'])

    # cpu_offload: CLI --image-cpu-offload > yaml > default
    # For store_true flags, check if explicitly set (True means it was passed)
    if shared.args.image_cpu_offload:
        effective['cpu_offload'] = True
    else:
        effective['cpu_offload'] = yaml_settings.get('cpu_offload', DEFAULTS['cpu_offload'])

    # compile_model: CLI --image-compile > yaml > default
    if shared.args.image_compile:
        effective['compile_model'] = True
    else:
        effective['compile_model'] = yaml_settings.get('compile_model', DEFAULTS['compile_model'])

    return effective


def save_image_model_settings(model_name, dtype, attn_backend, cpu_offload, compile_model):
    """Save image model settings to yaml."""
    settings_path = get_settings_path()

    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {
        'model_name': model_name,
        'dtype': dtype,
        'attn_backend': attn_backend,
        'cpu_offload': cpu_offload,
        'compile_model': compile_model,
    }

    try:
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)
        logger.info(f"Saved image model settings to {settings_path}")
    except Exception as e:
        logger.error(f"Failed to save image model settings: {e}")
