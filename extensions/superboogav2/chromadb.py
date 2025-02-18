import math
import random
import threading
import torch
import chromadb
import numpy as np
import posthog
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import extensions.superboogav2.parameters as parameters
from modules.logging_colors import logger
from modules.text_generation import decode, encode

# Intercept calls to posthog
posthog.capture = lambda *args, **kwargs: None


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
        else:  # Min is default
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


class ChromaCollector():
    def __init__(self):
        name = "".join(random.choice("ab") for _ in range(10))

        self.name = name
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            "sentence-transformers/all-mpnet-base-v2",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = chroma_client.create_collection(
            name=self.name,
            embedding_function=self.embedder,
            metadata={
                "hnsw:search_ef": 200,
                "hnsw:construction_ef": 200,
                "hnsw:M": 64,
            },
        )

        self.ids = []
        self.id_to_info = {}
        self.embeddings_cache = {}
        self.lock = threading.Lock()  # Locking so the server doesn't break.

    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int], metadatas: list[dict] = None):
        with self.lock:
            assert metadatas is None or len(metadatas) == len(texts), "metadatas must be None or have the same length as texts"

            if len(texts) == 0:
                return

            new_ids = self._get_new_ids(len(texts))

            (existing_texts, existing_embeddings, existing_ids, existing_metas), \
                (non_existing_texts, non_existing_ids, non_existing_metas) = self._split_texts_by_cache_hit(texts, new_ids, metadatas)

            # If there are any already existing texts, add them all at once.
            if existing_texts:
                logger.info(f'Adding {len(existing_embeddings)} cached embeddings.')
                args = {'embeddings': existing_embeddings, 'documents': existing_texts, 'ids': existing_ids}
                if metadatas is not None:
                    args['metadatas'] = existing_metas
                self.collection.add(**args)

            # If there are any non-existing texts, compute their embeddings all at once. Each call to embed has significant overhead.
            if non_existing_texts:
                non_existing_embeddings = self.embedder(non_existing_texts)
                for text, embedding in zip(non_existing_texts, non_existing_embeddings):
                    self.embeddings_cache[text] = embedding

                logger.info(f'Adding {len(non_existing_embeddings)} new embeddings.')
                args = {'embeddings': non_existing_embeddings, 'documents': non_existing_texts, 'ids': non_existing_ids}
                if metadatas is not None:
                    args['metadatas'] = non_existing_metas
                self.collection.add(**args)

            # Create a dictionary that maps each ID to its context and starting index
            new_info = {
                id_: {'text_with_context': context, 'start_index': start_index}
                for id_, context, start_index in zip(new_ids, texts_with_context, starting_indices)
            }

            self.id_to_info.update(new_info)
            self.ids.extend(new_ids)

    def _split_texts_by_cache_hit(self, texts: list[str], new_ids: list[str], metadatas: list[dict]):
        existing_texts, non_existing_texts = [], []
        existing_embeddings = []
        existing_ids, non_existing_ids = [], []
        existing_metas, non_existing_metas = [], []

        for i, text in enumerate(texts):
            id_ = new_ids[i]
            metadata = metadatas[i] if metadatas is not None else None
            embedding = self.embeddings_cache.get(text)
            if embedding is not None and embedding.any():
                existing_texts.append(text)
                existing_embeddings.append(embedding)
                existing_ids.append(id_)
                existing_metas.append(metadata)
            else:
                non_existing_texts.append(text)
                non_existing_ids.append(id_)
                non_existing_metas.append(metadata)

        return (existing_texts, existing_embeddings, existing_ids, existing_metas), \
               (non_existing_texts, non_existing_ids, non_existing_metas)

    def _get_new_ids(self, num_new_ids: int):
        if self.ids:
            max_existing_id = max(int(id_) for id_ in self.ids)
        else:
            max_existing_id = -1

        return [str(i + max_existing_id + 1) for i in range(num_new_ids)]

    def _find_min_max_start_index(self):
        max_index, min_index = 0, float('inf')
        for _, val in self.id_to_info.items():
            if val['start_index'] > max_index:
                max_index = val['start_index']
            if val['start_index'] < min_index:
                min_index = val['start_index']
        return min_index, max_index

    # NB: Does not make sense to weigh excerpts from different documents.
    # But let's say that's the user's problem. Perfect world scenario:
    # Apply time weighing to different documents. For each document, then, add
    # separate time weighing.

    def _apply_sigmoid_time_weighing(self, infos: list[Info], document_len: int, time_steepness: float, time_power: float):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        weights = sigmoid(time_steepness * np.linspace(-10, 10, document_len))

        # Scale to [0,time_power] and shift it up to [1-time_power, 1]
        weights = weights - min(weights)
        weights = weights * (time_power / max(weights))
        weights = weights + (1 - time_power)

        # Reverse the weights
        weights = weights[::-1]

        for info in infos:
            index = info.start_index
            info.distance *= weights[index]

    def _filter_outliers_by_median_distance(self, infos: list[Info], significant_level: float):
        # Ensure there are infos to filter
        if not infos:
            return []

        # Find info with minimum distance
        min_info = min(infos, key=lambda x: x.distance)

        # Calculate median distance among infos
        median_distance = np.median([inf.distance for inf in infos])

        # Filter out infos that have a distance significantly greater than the median
        filtered_infos = [inf for inf in infos if inf.distance <= significant_level * median_distance]

        # Always include the info with minimum distance
        if min_info not in filtered_infos:
            filtered_infos.append(min_info)

        return filtered_infos

    def _merge_infos(self, infos: list[Info]):
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

    # Main function for retrieving chunks by distance. It performs merging, time weighing, and mean filtering.

    def _get_documents_ids_distances(self, search_strings: list[str], n_results: int):
        n_results = min(len(self.ids), n_results)
        if n_results == 0:
            return [], [], []

        if isinstance(search_strings, str):
            search_strings = [search_strings]

        infos = []
        min_start_index, max_start_index = self._find_min_max_start_index()

        for search_string in search_strings:
            result = self.collection.query(query_texts=search_string, n_results=math.ceil(n_results / len(search_strings)), include=['distances'])
            curr_infos = [Info(start_index=self.id_to_info[id]['start_index'],
                               text_with_context=self.id_to_info[id]['text_with_context'],
                               distance=distance, id=id)
                          for id, distance in zip(result['ids'][0], result['distances'][0])]

            self._apply_sigmoid_time_weighing(infos=curr_infos, document_len=max_start_index - min_start_index + 1, time_steepness=parameters.get_time_steepness(), time_power=parameters.get_time_power())
            curr_infos = self._filter_outliers_by_median_distance(curr_infos, parameters.get_significant_level())
            infos.extend(curr_infos)

        infos.sort(key=lambda x: x.start_index)
        infos = self._merge_infos(infos)

        texts_with_context = [inf.text_with_context for inf in infos]
        ids = [inf.id for inf in infos]
        distances = [inf.distance for inf in infos]

        return texts_with_context, ids, distances

    # Get chunks by similarity

    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        with self.lock:
            documents, _, _ = self._get_documents_ids_distances(search_strings, n_results)
            return documents

    # Get ids by similarity

    def get_ids(self, search_strings: list[str], n_results: int) -> list[str]:
        with self.lock:
            _, ids, _ = self._get_documents_ids_distances(search_strings, n_results)
            return ids

    # Cutoff token count

    def _get_documents_up_to_token_count(self, documents: list[str], max_token_count: int):
        # TODO: Move to caller; We add delimiters there which might go over the limit.
        current_token_count = 0
        return_documents = []

        for doc in documents:
            doc_tokens = encode(doc)[0]
            doc_token_count = len(doc_tokens)
            if current_token_count + doc_token_count > max_token_count:
                # If adding this document would exceed the max token count,
                # truncate the document to fit within the limit.
                remaining_tokens = max_token_count - current_token_count

                truncated_doc = decode(doc_tokens[:remaining_tokens], skip_special_tokens=True)
                return_documents.append(truncated_doc)
                break
            else:
                return_documents.append(doc)
                current_token_count += doc_token_count

        return return_documents

    # Get chunks by similarity and then sort by ids

    def get_sorted_by_ids(self, search_strings: list[str], n_results: int, max_token_count: int) -> list[str]:
        with self.lock:
            documents, ids, _ = self._get_documents_ids_distances(search_strings, n_results)
            sorted_docs = [x for _, x in sorted(zip(ids, documents))]

            return self._get_documents_up_to_token_count(sorted_docs, max_token_count)

    # Get chunks by similarity and then sort by distance (lowest distance is last).

    def get_sorted_by_dist(self, search_strings: list[str], n_results: int, max_token_count: int) -> list[str]:
        with self.lock:
            documents, _, distances = self._get_documents_ids_distances(search_strings, n_results)
            sorted_docs = [doc for doc, _ in sorted(zip(documents, distances), key=lambda x: x[1])]  # sorted lowest -> highest

            # If a document is truncated or competely skipped, it would be with high distance.
            return_documents = self._get_documents_up_to_token_count(sorted_docs, max_token_count)
            return_documents.reverse()  # highest -> lowest

            return return_documents

    def delete(self, ids_to_delete: list[str], where: dict):
        with self.lock:
            ids_to_delete = self.collection.get(ids=ids_to_delete, where=where)['ids']
            if not ids_to_delete:
                return
            self.collection.delete(ids=ids_to_delete, where=where)

            # Remove the deleted ids from self.ids and self.id_to_info
            ids_set = set(ids_to_delete)
            self.ids = [id_ for id_ in self.ids if id_ not in ids_set]
            for id_ in ids_to_delete:
                self.id_to_info.pop(id_, None)

            logger.info(f'Successfully deleted {len(ids_to_delete)} records from chromaDB.')

    def clear(self):
        with self.lock:
            self.__init__()  # reinitialize the collector
            logger.info('Successfully cleared all records and reset chromaDB.')


def make_collector():
    return ChromaCollector()
