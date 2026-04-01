import time

import modules.shared as shared
from modules.logging_colors import logger
from modules.utils import resolve_model_path


def get_quantization_config(quant_method):
    """
    Get the appropriate quantization config based on the selected method.
    Applies quantization to both the transformer and the text_encoder.
    """
    import torch
    # Import BitsAndBytesConfig from BOTH libraries to be safe
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from diffusers import TorchAoConfig
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnBConfig

    if quant_method == 'none' or not quant_method:
        return None

    # Bitsandbytes 8-bit quantization
    elif quant_method == 'bnb-8bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersBnBConfig(
                    load_in_8bit=True
                ),
                "text_encoder": TransformersBnBConfig(
                    load_in_8bit=True
                )
            }
        )

    # Bitsandbytes 4-bit quantization
    elif quant_method == 'bnb-4bit':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                ),
                "text_encoder": TransformersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
            }
        )

    # torchao int8 weight-only
    elif quant_method == 'torchao-int8wo':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig("int8wo"),
                "text_encoder": TorchAoConfig("int8wo")
            }
        )

    # torchao fp4 (e2m1)
    elif quant_method == 'torchao-fp4':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig("fp4_e2m1"),
                "text_encoder": TorchAoConfig("fp4_e2m1")
            }
        )

    # torchao float8 weight-only
    elif quant_method == 'torchao-float8wo':
        return PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig("float8wo"),
                "text_encoder": TorchAoConfig("float8wo")
            }
        )

    else:
        logger.warning(f"Unknown quantization method: {quant_method}. Loading without quantization.")
        return None


def get_pipeline_type(pipe):
    """
    Detect the pipeline type based on the loaded pipeline class.

    Returns:
        str: 'zimage', 'qwenimage', or 'unknown'
    """
    class_name = pipe.__class__.__name__
    if class_name == 'ZImagePipeline':
        return 'zimage'
    elif class_name == 'QwenImagePipeline':
        return 'qwenimage'
    else:
        return 'unknown'


def load_image_model(model_name, dtype='bfloat16', attn_backend='sdpa', cpu_offload=False, compile_model=False, quant_method='none'):
    """
    Load a diffusers image generation model.

    Args:
        model_name: Name of the model directory
        dtype: 'bfloat16' or 'float16'
        attn_backend: 'sdpa' or 'flash_attention_2'
        cpu_offload: Enable CPU offloading for low VRAM
        compile_model: Compile the model for faster inference (slow first run)
        quant_method: 'none', 'bnb-8bit', 'bnb-4bit', or torchao options (int8wo, fp4, float8wo)
    """
    import torch
    from diffusers import DiffusionPipeline

    from modules.torch_utils import get_device

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

        # Use DiffusionPipeline for automatic pipeline detection
        # This handles both ZImagePipeline and QwenImagePipeline
        pipe = DiffusionPipeline.from_pretrained(
            str(model_path),
            **load_kwargs
        )

        pipeline_type = get_pipeline_type(pipe)

        if not cpu_offload:
            pipe.to(get_device())

        modules = ["transformer", "unet"]

        # Set attention backend
        if attn_backend == 'flash_attention_2':
            for name in modules:
                mod = getattr(pipe, name, None)
                if hasattr(mod, "set_attention_backend"):
                    mod.set_attention_backend("flash")
                    break

        # Compile model
        if compile_model:
            for name in modules:
                mod = getattr(pipe, name, None)
                if hasattr(mod, "compile"):
                    logger.info("Compiling model (first run will be slow)...")
                    mod.compile()
                    break

        if cpu_offload:
            pipe.enable_model_cpu_offload()

        shared.image_model = pipe
        shared.image_model_name = model_name
        shared.image_pipeline_type = pipeline_type

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
    shared.image_pipeline_type = None

    from modules.torch_utils import clear_torch_cache
    clear_torch_cache()

    logger.info("Image model unloaded.")
