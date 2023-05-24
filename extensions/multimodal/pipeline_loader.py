import traceback
from importlib import import_module
from pathlib import Path
from typing import Tuple

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger


def _get_available_pipeline_modules():
    pipeline_path = Path(__file__).parent / 'pipelines'
    modules = [p for p in pipeline_path.iterdir() if p.is_dir()]
    return [m.name for m in modules if (m / 'pipelines.py').exists()]


def load_pipeline(params: dict) -> Tuple[AbstractMultimodalPipeline, str]:
    pipeline_modules = {}
    available_pipeline_modules = _get_available_pipeline_modules()
    for name in available_pipeline_modules:
        try:
            pipeline_modules[name] = import_module(f'extensions.multimodal.pipelines.{name}.pipelines')
        except:
            logger.warning(f'Failed to get multimodal pipelines from {name}')
            logger.warning(traceback.format_exc())

    if shared.args.multimodal_pipeline is not None:
        for k in pipeline_modules:
            if hasattr(pipeline_modules[k], 'get_pipeline'):
                pipeline = getattr(pipeline_modules[k], 'get_pipeline')(shared.args.multimodal_pipeline, params)
                if pipeline is not None:
                    return (pipeline, k)
    else:
        model_name = shared.args.model.lower()
        for k in pipeline_modules:
            if hasattr(pipeline_modules[k], 'get_pipeline_from_model_name'):
                pipeline = getattr(pipeline_modules[k], 'get_pipeline_from_model_name')(model_name, params)
                if pipeline is not None:
                    return (pipeline, k)

    available = []
    for k in pipeline_modules:
        if hasattr(pipeline_modules[k], 'available_pipelines'):
            pipelines = getattr(pipeline_modules[k], 'available_pipelines')
            available += pipelines

    if shared.args.multimodal_pipeline is not None:
        log = f'Multimodal - ERROR: Failed to load multimodal pipeline "{shared.args.multimodal_pipeline}", available pipelines are: {available}.'
    else:
        log = f'Multimodal - ERROR: Failed to determine multimodal pipeline for model {shared.args.model}, please select one manually using --multimodal-pipeline [PIPELINE]. Available pipelines are: {available}.'
    logger.critical(f'{log} Please specify a correct pipeline, or disable the extension')
    raise RuntimeError(f'{log} Please specify a correct pipeline, or disable the extension')
