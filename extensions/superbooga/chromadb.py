from typing import Optional

import chromadb
import posthog
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from modules.logging_colors import logger

logger.info('Intercepting all calls to posthog :)')
posthog.capture = lambda *args, **kwargs: None


class Collecter():
    def __init__(self):
        pass

    def add(self, texts: list[str]):
        pass

    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        pass

    def clear(self):
        pass


class Embedder():
    DEFAULT_DOCUMENT_TEMPLATE = '<|text|>'
    DEFAULT_QUERY_TEMPLATE = '<|text|>'

    def __init__(self, document_template: Optional[str] = None, query_template: Optional[str] = None):
        self.document_template = document_template or self.DEFAULT_DOCUMENT_TEMPLATE
        self.query_template = query_template or self.DEFAULT_QUERY_TEMPLATE

    def embed(self, text: str) -> list[torch.Tensor]:
        pass

    def embed_document(self, text: str) -> list[torch.Tensor]:
        pass

    def embed_query(self, text: str) -> list[torch.Tensor]:
        pass


class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = embedder
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=embedder.embed)
        self.ids = []

    def add(self, texts: list[str]):
        if len(texts) == 0:
            return

        self.ids = [f"id{i}" for i in range(len(texts))]
        embeddings = self.embedder.embed_document(texts)
        self.collection.add(documents=texts, embeddings=embeddings, ids=self.ids)

    def get_documents_and_ids(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        if n_results == 0:
            return [], []

        search_embeddings = self.embedder.embed_query(search_strings)
        result = self.collection.query(query_embeddings=search_embeddings, n_results=n_results, include=['documents'])
        documents = result['documents'][0]
        ids = list(map(lambda x: int(x[2:]), result['ids'][0]))
        return documents, ids

    # Get chunks by similarity
    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, _ = self.get_documents_and_ids(search_strings, n_results)
        return documents

    # Get ids by similarity
    def get_ids(self, search_strings: list[str], n_results: int) -> list[str]:
        _, ids = self.get_documents_and_ids(search_strings, n_results)
        return ids

    # Get chunks by similarity and then sort by insertion order
    def get_sorted(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, ids = self.get_documents_and_ids(search_strings, n_results)
        return [x for _, x in sorted(zip(ids, documents))]

    # Get ids by similarity and then sort by insertion order
    def get_ids_sorted(self, search_strings: list[str], n_results: int) -> list[str]:
        _, ids = self.get_documents_and_ids(search_strings, n_results)
        return sorted(ids)

    def clear(self):
        self.collection.delete(ids=self.ids)
        self.ids = []


class SentenceTransformerEmbedder(Embedder):
    DEFAULT_MODEL_NAME_OR_PATH = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_name_or_path: Optional[str] = None, document_template: Optional[str] = None, query_template: Optional[str] = None) -> None:
        super().__init__(document_template=document_template, query_template=query_template)
        self.model = SentenceTransformer(model_name_or_path or self.DEFAULT_MODEL_NAME_OR_PATH)
        self.embed = self.model.encode

    def embed_document(self, text: str):
        if isinstance(text, str):
            text = [text]
        text = [self.document_template.replace('<|text|>', t) for t in text]
        return list(self.model.encode(text))

    def embed_query(self, text: str):
        if isinstance(text, str):
            text = [text]
        text = [self.query_template.replace('<|text|>', t) for t in text]
        return list(self.model.encode(text))


try:
    from InstructorEmbedding import INSTRUCTOR
    class InstructorEmbedder(Embedder):
        DEFAULT_MODEL_NAME_OR_PATH = "hkunlp/instructor-base"

        def __init__(self, model_name_or_path: Optional[str] = None, document_template: Optional[str] = None, query_template: Optional[str] = None) -> None:
            super().__init__(document_template=document_template, query_template=query_template)
            self.model = INSTRUCTOR(model_name_or_path or self.DEFAULT_MODEL_NAME_OR_PATH)
            self.embed = self.model.encode

        def embed_document(self, text: str):
            if isinstance(text, str):
                text = [text]
            text = [self.document_template.replace('<|text|>', t) for t in text]
            return list(self.model.encode(text))

        def embed_query(self, text: str):
            if isinstance(text, str):
                text = [text]
            text = [self.query_template.replace('<|text|>', t) for t in text]
            return list(self.model.encode(text))
except ImportError:
    pass


def get_default_embedder() -> Embedder:
    global embedder_default
    if not embedder_default:
        embedder_default = SentenceTransformerEmbedder()
    return embedder_default


def make_embedder(model_type: Optional[str] = None, **kwargs) -> Embedder:
    if not model_type:
        return get_default_embedder()
    elif model_type == 'sentence_transformer':
        return SentenceTransformerEmbedder(**kwargs)
    elif model_type == 'instructor':
        return InstructorEmbedder(**kwargs)
    else:
        raise ValueError("Unknown embedder model type specified. Only 'sentence_transformer' and 'instructor' are supported")


def make_collector(embedder: Optional[Embedder] = None):
    return ChromaCollector(embedder or get_default_embedder())


def add_chunks_to_collector(chunks, collector):
    collector.clear()
    collector.add(chunks)


embedder = SentenceTransformerEmbedder()
