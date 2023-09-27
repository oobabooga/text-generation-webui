import os

import numpy as np
from extensions.openai.errors import ServiceUnavailableError
from extensions.openai.utils import debug_msg, float_list_to_base64
from sentence_transformers import SentenceTransformer

embeddings_params_initialized = False
# using 'lazy loading' to avoid circular import
# so this function will be executed only once
def initialize_embedding_params():
    global embeddings_params_initialized
    if not embeddings_params_initialized:
        global st_model, embeddings_model, embeddings_device
        from extensions.openai.script import params
        st_model = os.environ.get("OPENEDAI_EMBEDDING_MODEL", params.get('embedding_model', 'all-mpnet-base-v2'))
        embeddings_model = None
        # OPENEDAI_EMBEDDING_DEVICE: auto (best or cpu), cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
        embeddings_device = os.environ.get("OPENEDAI_EMBEDDING_DEVICE", params.get('embedding_device', 'cpu'))
        if embeddings_device.lower() == 'auto':
            embeddings_device = None
        embeddings_params_initialized = True


def load_embedding_model(model: str) -> SentenceTransformer:
    initialize_embedding_params()
    global embeddings_device, embeddings_model
    try:
        embeddings_model = 'loading...'  # flag
        # see: https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer
        emb_model = SentenceTransformer(model, device=embeddings_device)
        # ... emb_model.device doesn't seem to work, always cpu anyways? but specify cpu anyways to free more VRAM
        print(f"\nLoaded embedding model: {model} on {emb_model.device} [always seems to say 'cpu', even if 'cuda'], max sequence length: {emb_model.max_seq_length}")
    except Exception as e:
        embeddings_model = None
        raise ServiceUnavailableError(f"Error: Failed to load embedding model: {model}", internal_message=repr(e))

    return emb_model


def get_embeddings_model() -> SentenceTransformer:
    initialize_embedding_params()
    global embeddings_model, st_model
    if st_model and not embeddings_model:
        embeddings_model = load_embedding_model(st_model)  # lazy load the model
    return embeddings_model


def get_embeddings_model_name() -> str:
    initialize_embedding_params()
    global st_model
    return st_model


def get_embeddings(input: list) -> np.ndarray:
    return get_embeddings_model().encode(input, convert_to_numpy=True, normalize_embeddings=True, convert_to_tensor=False, device=embeddings_device)


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
