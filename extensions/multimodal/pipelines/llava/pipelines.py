from typing import Optional

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline

available_pipelines = ['llava-7b', 'llava-13b', 'llava-llama-2-13b']


def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if name == 'llava-7b':
        from .llava import LLaVA_v0_7B_Pipeline
        return LLaVA_v0_7B_Pipeline(params)
    if name == 'llava-13b':
        from .llava import LLaVA_v0_13B_Pipeline
        return LLaVA_v0_13B_Pipeline(params)
    if name == 'llava-llama-2-13b':
        from .llava import LLaVA_LLaMA_2_13B_Pipeline
        return LLaVA_LLaMA_2_13B_Pipeline(params)
    return None


def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if 'llava' not in model_name.lower():
        return None
    if 'llama-2' in model_name.lower():
        if '13b' in model_name.lower():
            from .llava import LLaVA_LLaMA_2_13B_Pipeline
            return LLaVA_LLaMA_2_13B_Pipeline(params)
    if '7b' in model_name.lower():
        from .llava import LLaVA_v0_7B_Pipeline
        return LLaVA_v0_7B_Pipeline(params)
    if '13b' in model_name.lower():
        from .llava import LLaVA_v0_13B_Pipeline
        return LLaVA_v0_13B_Pipeline(params)
    return None
