import sys
import time

import modules.shared as shared
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata
from modules.utils import resolve_model_path

last_generation_time = time.time()


def load_model(model_name, loader=None):
    logger.info(f"Loading \"{model_name}\"")
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    load_func_map = {
        'llama.cpp': llama_cpp_server_loader,
        'Transformers': transformers_loader,
        'ExLlamav3_HF': ExLlamav3_HF_loader,
        'ExLlamav3': ExLlamav3_loader,
        'ExLlamav2_HF': ExLlamav2_HF_loader,
        'ExLlamav2': ExLlamav2_loader,
        'TensorRT-LLM': TensorRT_LLM_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('The path to the model does not exist. Exiting.')
                raise ValueError

    if loader != 'llama.cpp' and 'sampler_hijack' not in sys.modules:
        from modules import sampler_hijack
        sampler_hijack.hijack_samplers()

    shared.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is not None:
            from modules.transformers_loader import load_tokenizer
            tokenizer = load_tokenizer(model_name)

    if model is None:
        return None, None

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    if loader.lower().startswith('exllama') or loader.lower().startswith('tensorrt') or loader == 'llama.cpp':
        shared.settings['truncation_length'] = shared.args.ctx_size

    shared.is_multimodal = False
    if loader.lower() in ('exllamav3', 'llama.cpp') and hasattr(model, 'is_multimodal'):
        shared.is_multimodal = model.is_multimodal()

    logger.info(f"Loaded \"{model_name}\" in {(time.time()-t0):.2f} seconds.")
    logger.info(f"LOADER: \"{loader}\"")
    logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: \"{metadata['instruction_template']}\"")
    return model, tokenizer


def llama_cpp_server_loader(model_name):
    from modules.llama_cpp_server import LlamaServer

    path = resolve_model_path(model_name)

    if path.is_file():
        model_file = path
    else:
        gguf_files = sorted(path.glob('*.gguf'))
        if not gguf_files:
            logger.error(f"No .gguf models found in the directory: {path}")
            return None, None

        model_file = gguf_files[0]

    try:
        model = LlamaServer(model_file)
        return model, model
    except Exception as e:
        logger.error(f"Error loading the model with llama.cpp: {str(e)}")
        return None, None


def transformers_loader(model_name):
    from modules.transformers_loader import load_model_HF
    return load_model_HF(model_name)


def ExLlamav3_HF_loader(model_name):
    from modules.exllamav3_hf import Exllamav3HF

    return Exllamav3HF.from_pretrained(model_name)


def ExLlamav3_loader(model_name):
    from modules.exllamav3 import Exllamav3Model

    model, tokenizer = Exllamav3Model.from_pretrained(model_name)
    return model, tokenizer


def ExLlamav2_HF_loader(model_name):
    from modules.exllamav2_hf import Exllamav2HF

    return Exllamav2HF.from_pretrained(model_name)


def ExLlamav2_loader(model_name):
    from modules.exllamav2 import Exllamav2Model

    model, tokenizer = Exllamav2Model.from_pretrained(model_name)
    return model, tokenizer


def TensorRT_LLM_loader(model_name):
    try:
        from modules.tensorrt_llm import TensorRTLLMModel
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Failed to import 'tensorrt_llm'. Please install it manually following the instructions in the TensorRT-LLM GitHub repository.")

    model = TensorRTLLMModel.from_pretrained(model_name)
    return model


def unload_model(keep_model_name=False):
    if shared.model is None:
        return

    model_class_name = shared.model.__class__.__name__
    is_llamacpp = (model_class_name == 'LlamaServer')

    if model_class_name in ['Exllamav3Model', 'Exllamav3HF']:
        shared.model.unload()
    elif model_class_name in ['Exllamav2Model', 'Exllamav2HF'] and hasattr(shared.model, 'unload'):
        shared.model.unload()

    shared.model = shared.tokenizer = None
    shared.lora_names = []
    shared.model_dirty_from_training = False

    if not is_llamacpp:
        from modules.torch_utils import clear_torch_cache
        clear_torch_cache()

    if not keep_model_name:
        shared.model_name = 'None'


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def unload_model_if_idle():
    global last_generation_time

    logger.info(f"Setting a timeout of {shared.args.idle_timeout} minutes to unload the model in case of inactivity.")

    while True:
        shared.generation_lock.acquire()
        try:
            if time.time() - last_generation_time > shared.args.idle_timeout * 60:
                if shared.model is not None:
                    logger.info("Unloading the model for inactivity.")
                    unload_model(keep_model_name=True)
        finally:
            shared.generation_lock.release()

        time.sleep(60)
