import os

import numpy as np
from transformers import AutoModel

from extensions.openai.errors import ServiceUnavailableError
from extensions.openai.utils import debug_msg, float_list_to_base64
from modules.logging_colors import logger

embeddings_params_initialized = False


def initialize_embedding_params():
    '''
    using 'lazy loading' to avoid circular import
    so this function will be executed only once
    '''
    global embeddings_params_initialized
    if not embeddings_params_initialized:
        from extensions.openai.script import params

        global st_model, embeddings_model, embeddings_device

        st_model = os.environ.get("OPENEDAI_EMBEDDING_MODEL", params.get('embedding_model', 'all-mpnet-base-v2'))
        embeddings_model = None
        # OPENEDAI_EMBEDDING_DEVICE: auto (best or cpu), cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
        embeddings_device = os.environ.get("OPENEDAI_EMBEDDING_DEVICE", params.get('embedding_device', 'cpu'))
        if embeddings_device.lower() == 'auto':
            embeddings_device = None

        embeddings_params_initialized = True


def load_embedding_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        logger.error("The sentence_transformers module has not been found. Please install it manually with pip install -U sentence-transformers.")
        raise ModuleNotFoundError

    initialize_embedding_params()
    global embeddings_device, embeddings_model
    try:
        print(f"Try embedding model: {model} on {embeddings_device}")
        if 'jina-embeddings' in model:
            embeddings_model = AutoModel.from_pretrained(model, trust_remote_code=True)  # trust_remote_code is needed to use the encode method
            embeddings_model = embeddings_model.to(embeddings_device)
        else:
            embeddings_model = SentenceTransformer(model, device=embeddings_device)

        print(f"Loaded embedding model: {model}")
    except Exception as e:
        embeddings_model = None
        raise ServiceUnavailableError(f"Error: Failed to load embedding model: {model}", internal_message=repr(e))


def get_embeddings_model():
    initialize_embedding_params()
    global embeddings_model, st_model
    if st_model and not embeddings_model:
        load_embedding_model(st_model)  # lazy load the model

    return embeddings_model


def get_embeddings_model_name() -> str:
    initialize_embedding_params()
    global st_model
    return st_model


def get_embeddings(input: list) -> np.ndarray:
    model = get_embeddings_model()
    debug_msg(f"embedding model : {model}")
    embedding = model.encode(input, convert_to_numpy=True, normalize_embeddings=True, convert_to_tensor=False)
    debug_msg(f"embedding result : {embedding}")  # might be too long even for debug, use at you own will
    return embedding


def embeddings(input: list, encoding_format: str) -> dict:
    embeddings = get_embeddings(input)
    if encoding_format == "base64":
        data = [{"object": "embedding", "embedding": float_list_to_base64(emb), "index": n} for n, emb in enumerate(embeddings)]
    else:
        data = [{"object": "embedding", "embedding": emb.tolist(), "index": n} for n, emb in enumerate(embeddings)]

    response = {
        "object": "list",
        "data": data,
        "model": st_model,  # return the real model
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }

    debug_msg(f"Embeddings return size: {len(embeddings[0])}, number: {len(embeddings)}")
    return response
