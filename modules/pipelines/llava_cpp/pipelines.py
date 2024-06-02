from typing import Optional

from modules.abstract_pipeline import AbstractMultimodalPipeline

available_pipelines = ['Llava15ChatHandler']


def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if name == 'Llava15ChatHandler':
        from .llava import LLaVA_Cpp_Pipeline
        return LLaVA_Cpp_Pipeline(params)
    return None


def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if 'llava' not in model_name.lower():
        return None
    else:
        from .llava import LLaVA_Cpp_Pipeline
        return LLaVA_Cpp_Pipeline(params)
    return None
