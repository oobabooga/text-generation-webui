import os
from sentence_transformers import SentenceTransformer
from extensions.openai.utils import float_list_to_base64, debug_msg
from extensions.openai.errors import *

st_model = os.environ["OPENEDAI_EMBEDDING_MODEL"] if "OPENEDAI_EMBEDDING_MODEL" in os.environ else "all-mpnet-base-v2"
embeddings_model = None


def load_embedding_model(model):
    try:
        emb_model = SentenceTransformer(model)
        print(f"\nLoaded embedding model: {model}, max sequence length: {emb_model.max_seq_length}")
    except Exception as e:
        print(f"\nError: Failed to load embedding model: {model}")
        raise ServiceUnavailableError(f"Error: Failed to load embedding model: {model}", internal_message=repr(e))

    return emb_model


def get_embeddings_model():
    global embeddings_model, st_model
    if st_model and not embeddings_model:
        embeddings_model = load_embedding_model(st_model)  # lazy load the model
    return embeddings_model


def get_embeddings_model_name():
    global st_model
    return st_model


def embeddings(input: list, encoding_format: str):

    embeddings = get_embeddings_model().encode(input).tolist()

    if encoding_format == "base64":
        data = [{"object": "embedding", "embedding": float_list_to_base64(emb), "index": n} for n, emb in enumerate(embeddings)]
    else:
        data = [{"object": "embedding", "embedding": emb, "index": n} for n, emb in enumerate(embeddings)]

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
