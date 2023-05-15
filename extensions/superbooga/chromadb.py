import logging

import posthog
import torch
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

logging.info('Intercepting all calls to posthog :)')
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
    def __init__(self):
        pass

    def embed(self, text: str) -> list[torch.Tensor]:
        pass


class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = embedder
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=embedder.embed)
        self.ids = []

    def add(self, texts: list[str]):
        self.ids = [f"id{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, ids=self.ids)

    def get_documents_and_ids(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents'])
        documents = result['documents'][0]
        ids = list(map(lambda x: int(x[2:]), result['ids'][0]))
        return documents, ids

    def get_documents_and_ids_and_distances(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents', 'distances'])
        documents = result['documents'][0]
        ids = list(map(lambda x: int(x[2:]), result['ids'][0]))
        distances = result['distances'][0]
        return documents, ids, distances

    # Get chunks by similarity
    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, _ = self.get_documents_and_ids(search_strings, n_results)
        return documents

    # Get ids by similarity
    def get_ids(self, search_strings: list[str], n_results: int) -> list[str]:
        _ , ids = self.get_documents_and_ids(search_strings, n_results)
        return ids

    # Get chunks by similarity and then sort by insertion order
    def get_sorted(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, ids = self.get_documents_and_ids(search_strings, n_results)
        return [x for _, x in sorted(zip(ids, documents))]

    # Multiply distance by factor within [min_time_weight, 1] where more recent is lower
    def apply_time_weight_to_distances(self, ids: list[int], distances: list[float], min_time_weight: float = 1.0):
        if len(self.ids) <= 1: return distances.copy()
        return [distance * (_id / (len(self.ids)-1) * min_time_weight - _id / (len(self.ids)-1) + 1) for _id, distance in zip(ids, distances)]

    # Get ids by similarity and then sort by insertion order
    def get_ids_sorted(self, search_strings: list[str], n_results: int, n_initial: int = None, min_time_weight: float = 1.0) -> list[str]:
        do_time_weight = min_time_weight != 1
        if not (do_time_weight and n_initial):
            n_initial = n_results
        if n_initial < n_results:
            raise ValueError(f"n_initial {n_initial} should be >= n_results {n_results}")
        _ , ids , distances = self.get_documents_and_ids_and_distances(search_strings, n_initial)
        if do_time_weight:
            distances_w = self.apply_time_weight_to_distances(ids, distances, min_time_weight=min_time_weight)
            results = zip(ids, distances, distances_w)
            results = sorted(results, key = lambda x: x[2])[:n_results]
            results = sorted(results, key = lambda x: x[0])
            ids = [x[0] for x in results]
        return sorted(ids)

    def clear(self):
        self.collection.delete(ids=self.ids)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embed = self.model.encode


def make_collector():
    global embedder
    return ChromaCollector(embedder)


def add_chunks_to_collector(chunks, collector):
    collector.clear()
    collector.add(chunks)


embedder = SentenceTransformerEmbedder()
