import chromadb
import posthog
import torch
import numpy
import math

import extensions.superbooga.parameters as parameters

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from modules.logging_colors import logger

from modules import shared

logger.info('Intercepting all calls to posthog :)')
posthog.capture = lambda *args, **kwargs: None


class Collecter():
    def __init__(self):
        pass

    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int]):
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

class Info:
    def __init__(self, start_index, text_with_context, distance, id):
        self.text_with_context = text_with_context
        self.start_index = start_index
        self.distance = distance
        self.id = id

    def calculate_distance(self, other_info):
        if parameters.get_new_dist_strategy() == parameters.DIST_MIN_STRATEGY:
            # Min
            return min(self.distance, other_info.distance)
        elif parameters.get_new_dist_strategy() == parameters.DIST_HARMONIC_STRATEGY:
            # Harmonic mean
            return 2 * (self.distance * other_info.distance) / (self.distance + other_info.distance)
        elif parameters.get_new_dist_strategy() == parameters.DIST_GEOMETRIC_STRATEGY:
            # Geometric mean
            return (self.distance * other_info.distance) ** 0.5
        elif parameters.get_new_dist_strategy() == parameters.DIST_ARITHMETIC_STRATEGY:
            # Arithmetic mean
            return (self.distance + other_info.distance) / 2
        else: # Min is default
            return min(self.distance, other_info.distance)

    def merge_with(self, other_info):
        s1 = self.text_with_context
        s2 = other_info.text_with_context
        s1_start = self.start_index
        s2_start = other_info.start_index
        
        new_dist = self.calculate_distance(other_info)

        if self.should_merge(s1, s2, s1_start, s2_start):
            if s1_start <= s2_start:
                if s1_start + len(s1) >= s2_start + len(s2):  # if s1 completely covers s2
                    return Info(s1_start, s1, new_dist, self.id)
                else:
                    overlap = max(0, s1_start + len(s1) - s2_start)
                    return Info(s1_start, s1 + s2[overlap:], new_dist, self.id)
            else:
                if s2_start + len(s2) >= s1_start + len(s1):  # if s2 completely covers s1
                    return Info(s2_start, s2, new_dist, other_info.id)
                else:
                    overlap = max(0, s2_start + len(s2) - s1_start)
                    return Info(s2_start, s2 + s1[overlap:], new_dist, other_info.id)

        return None
    
    @staticmethod
    def should_merge(s1, s2, s1_start, s2_start):
        # Check if s1 and s2 are adjacent or overlapping
        s1_end = s1_start + len(s1)
        s2_end = s2_start + len(s2)
        
        return not (s1_end < s2_start or s2_end < s1_start)

class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = embedder
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=self.embedder.embed)
        self.ids = []
        self.embeddings_cache = {}

    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int]):
        if len(texts) == 0:
            return

        self.ids = [f"id{i}" for i in range(len(texts))]

        # Split the texts into existing and non-existing ones
        existing_texts = []
        non_existing_texts = []
        existing_embeddings = []

        for text, id_ in zip(texts, self.ids):
            embedding = self.embeddings_cache.get(text)
            if embedding:
                existing_texts.append((text, id_))
                existing_embeddings.append(embedding)
            else:
                non_existing_texts.append((text, id_))

        # If there are any already existing texts, add them all at once.
        if existing_texts:
            logger.info(f'Adding {len(existing_embeddings)} cached embeddings.')
            self.collection.add(embeddings=existing_embeddings, documents=[text for text, _ in existing_texts], ids=[id_ for _, id_ in existing_texts])

        # If there are any non-existing texts, compute their embeddings all at once. Each call to embed has significant overhead.
        if non_existing_texts:
            non_existing_embeddings = self.embedder.embed([text for text, _ in non_existing_texts]).tolist()
            for (text, _), embedding in zip(non_existing_texts, non_existing_embeddings):
                self.embeddings_cache[text] = embedding
            
            logger.info(f'Adding {len(non_existing_embeddings)} new embeddings.')
            self.collection.add(embeddings=non_existing_embeddings, documents=[text for text, _ in non_existing_texts], ids=[id_ for _, id_ in non_existing_texts])

        # Create a dictionary that maps each ID to its context and starting index
        self.id_to_info = {
            id_: {'text_with_context': context, 'start_index': start_index}
            for id_, context, start_index in zip(self.ids, texts_with_context, starting_indices)
        }


    def filter_infos_in_interval(self, infos: list[Info], percentage: float):
        points = [info.start_index for info in infos]
        weights = [1.0/info.distance for info in infos]

        start, end = self.find_optimal_weighted_interval(points, weights, percentage)

        return [info for info in infos if start <= info.start_index <= end]


    def find_optimal_weighted_interval(self, points: list[int], weights: list[float], percentage: float):
        total_weight = sum(weights)
        target_weight = total_weight * percentage

        weight_accumulator = 0
        left = 0
        min_length = float('inf')
        result = (None, None)

        for right in range(len(points)):
            weight_accumulator += weights[right]
            
            while weight_accumulator >= target_weight:
                if points[right] - points[left] < min_length:
                    min_length = points[right] - points[left]
                    result = (points[left], points[right])

                weight_accumulator -= weights[left]
                left += 1

        return result


    def merge_infos(self, infos: list[Info]):
        merged_infos = []
        current_info = infos[0]

        for next_info in infos[1:]:
            merged = current_info.merge_with(next_info)
            if merged is not None:
                current_info = merged
            else:
                merged_infos.append(current_info)
                current_info = next_info

        merged_infos.append(current_info)
        return merged_infos
    

    def get_documents_ids_distances(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        if n_results == 0:
            return [], [], []

        result = self.collection.query(query_texts=search_strings, n_results=math.ceil(n_results / len(search_strings)), include=['documents', 'distances'])
        infos = [Info(start_index=self.id_to_info[id]['start_index'], 
                    text_with_context=self.id_to_info[id]['text_with_context'], 
                    distance=distance, id=id) 
                for id, distance in zip(result['ids'][0], result['distances'][0])]
        
        print(result)

        infos.sort(key=lambda x: x.start_index)
        infos = self.filter_infos_in_interval(infos, parameters.get_confidence_interval())
        infos = self.merge_infos(infos)

        texts_with_context = [inf.text_with_context for inf in infos]
        ids = [inf.id for inf in infos]
        distances = [inf.distance for inf in infos]

        return texts_with_context, ids, distances

    # Get chunks by similarity
    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        documents, _, _ = self.get_documents_ids_distances(search_strings, n_results)
        return documents

    # Get ids by similarity
    def get_ids(self, search_strings: list[str], n_results: int) -> list[str]:
        _, ids, _ = self.get_documents_ids_distances(search_strings, n_results)
        return ids
    
    # Cutoff token count
    def get_documents_up_to_token_count(self, documents: list[str], max_token_count: int):
        # TODO: Move to script.py; We add delimiters there which might go over the limit.
        current_token_count = 0
        return_documents = []

        for doc in documents:
            doc_tokens = shared.tokenizer.encode(doc)
            doc_token_count = len(doc_tokens)
            if current_token_count + doc_token_count > max_token_count:
                # If adding this document would exceed the max token count,
                # truncate the document to fit within the limit.
                remaining_tokens = max_token_count - current_token_count
                
                truncated_doc = shared.tokenizer.decode(doc_tokens[:remaining_tokens], skip_special_tokens=True)
                return_documents.append(truncated_doc)
                break
            else:
                return_documents.append(doc)
                current_token_count += doc_token_count

        return return_documents

    # Get chunks by similarity and then sort by insertion order
    def get_sorted(self, search_strings: list[str], n_results: int, max_token_count: int) -> list[str]:
        documents, ids, _ = self.get_documents_ids_distances(search_strings, n_results)
        sorted_docs = [x for _, x in sorted(zip(ids, documents))]

        return self.get_documents_up_to_token_count(sorted_docs, max_token_count)
    
    # Get chunks by similarity and then sort by distance (lowest distance is last).
    def get_sorted_by_dist(self, search_strings: list[str], n_results: int, max_token_count: int) -> list[str]:
        documents, _, distances = self.get_documents_ids_distances(search_strings, n_results)
        sorted_docs = [doc for doc, _ in sorted(zip(documents, distances), key=lambda x: x[1])] # sorted lowest -> highest
        
        # If a document is truncated or competely skipped, it would be with high distance.
        return_documents = self.get_documents_up_to_token_count(sorted_docs, max_token_count)
        return_documents.reverse() # highest -> lowest

        return return_documents


    # Multiply distance by factor within [0, time_weight] where more recent is lower
    def apply_time_weight_to_distances(self, ids: list[int], distances: list[float], time_weight: float = 1.0) -> list[float]:
        if len(self.ids) <= 1:
            return distances.copy()

        return [distance * (1 - _id / (len(self.ids) - 1) * time_weight) for _id, distance in zip(ids, distances)]

    # Get ids by similarity and then sort by insertion order
    def get_ids_sorted(self, search_strings: list[str], n_results: int, n_initial: int = None, time_weight: float = 1.0) -> list[str]:
        do_time_weight = time_weight > 0
        if not (do_time_weight and n_initial is not None):
            n_initial = n_results
        elif n_initial == -1:
            n_initial = len(self.ids)

        if n_initial < n_results:
            raise ValueError(f"n_initial {n_initial} should be >= n_results {n_results}")

        _, ids, distances = self.get_documents_ids_distances(search_strings, n_initial)
        if do_time_weight:
            distances_w = self.apply_time_weight_to_distances(ids, distances, time_weight=time_weight)
            results = zip(ids, distances, distances_w)
            results = sorted(results, key=lambda x: x[2])[:n_results]
            results = sorted(results, key=lambda x: x[0])
            ids = [x[0] for x in results]

        return sorted(ids)

    def clear(self):
        self.chroma_client.delete_collection("context")
        self.collection = self.chroma_client.create_collection("context", embedding_function=self.embedder.embed)
        self.ids = []


class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embed = self.model.encode


def make_collector():
    global embedder
    return ChromaCollector(embedder)


def add_chunks_to_collector(data_chunks, data_chunks_with_context, data_chunk_starting_indices, collector):
    collector.clear()
    collector.add(data_chunks, data_chunks_with_context, data_chunk_starting_indices)


embedder = SentenceTransformerEmbedder()
