import time

import torch

import modules.shared as shared
from modules.logging_colors import logger
from modules.torch_utils import get_device
from modules.utils import resolve_model_path


def get_quantization_config(quant_method):
    """
    Get the appropriate quantization config based on the selected method.

    Args:
        quant_method: One of 'none', 'bnb-8bit', 'bnb-4bit', 'quanto-8bit', 'quanto-4bit', 'quanto-2bit'

    Returns:
        PipelineQuantizationConfig or None
    """
    from diffusers.quantizers import PipelineQuantizationConfig
    from diffusers import BitsAndBytesConfig, QuantoConfig

    if quant_method == 'none' or not quant_method:
        return None

    # Bitsandbytes 8-bit quantization
    elif quant_method == 'bnb-8bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": BitsAndBytesConfig(
                    load_in_8bit=True
                )
            }
        )

    # Bitsandbytes 4-bit quantization
    elif quant_method == 'bnb-4bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            }
        )

    # Quanto 8-bit quantization
    elif quant_method == 'quanto-8bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": QuantoConfig(weights_dtype="int8")
            }
        )

    # Quanto 4-bit quantization
    elif quant_method == 'quanto-4bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": QuantoConfig(weights_dtype="int4")
            }
        )

    # Quanto 2-bit quantization
    elif quant_method == 'quanto-2bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": QuantoConfig(weights_dtype="int2")
            }
        )

    else:
        logger.warning(f"Unknown quantization method: {quant_method}. Loading without quantization.")
        return None


def load_image_model(model_name, dtype='bfloat16', attn_backend='sdpa', cpu_offload=False, compile_model=False, quant_method='none'):
    """
    Load a diffusers image generation model.

    Args:
        model_name: Name of the model directory
        dtype: 'bfloat16' or 'float16'
        attn_backend: 'sdpa', 'flash_attention_2', or 'flash_attention_3'
        cpu_offload: Enable CPU offloading for low VRAM
        compile_model: Compile the model for faster inference (slow first run)
        quant_method: Quantization method - 'none', 'bnb-8bit', 'bnb-4bit', 'quanto-8bit', 'quanto-4bit', 'quanto-2bit'
    """
    from diffusers import ZImagePipeline

    logger.info(f"Loading image model \"{model_name}\" with quantization: {quant_method}")
    t0 = time.time()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    target_dtype = dtype_map.get(dtype, torch.bfloat16)

    model_path = resolve_model_path(model_name, image_model=True)

    try:
        # Get quantization config based on selected method
        pipeline_quant_config = get_quantization_config(quant_method)

        # Load the pipeline
        load_kwargs = {
            "torch_dtype": target_dtype,
            "low_cpu_mem_usage": True,
        }

        if pipeline_quant_config is not None:
            load_kwargs["quantization_config"] = pipeline_quant_config

        pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            **load_kwargs
        )

        if not cpu_offload:
            pipe.to(get_device())

        # Set attention backend
        if attn_backend == 'flash_attention_2':
            pipe.transformer.set_attention_backend("flash")
        elif attn_backend == 'flash_attention_3':
            pipe.transformer.set_attention_backend("_flash_3")
        # sdpa is the default, no action needed

        if compile_model:
            logger.info("Compiling model (first run will be slow)...")
            pipe.transformer.compile()

        if cpu_offload:
            pipe.enable_model_cpu_offload()

        shared.image_model = pipe
        shared.image_model_name = model_name

        logger.info(f"Loaded image model \"{model_name}\" in {(time.time() - t0):.2f} seconds.")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load image model: {str(e)}")
        return None


def unload_image_model():
    """Unload the current image model and free VRAM."""
    if shared.image_model is None:
        return

    del shared.image_model
    shared.image_model = None
    shared.image_model_name = 'None'

    from modules.torch_utils import clear_torch_cache
    clear_torch_cache()

    logger.info("Image model unloaded.")
