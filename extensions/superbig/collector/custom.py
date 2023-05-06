import posthog

print('Intercepting all calls to posthog :)')
posthog.capture = lambda *args, **kwargs: None

import chromadb
from chromadb.api import Collection
from chromadb.config import Settings
from ..base import Chunk, Collecter, Embedder, Bucket, InjectionPoint

class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = embedder
        self.collections: dict[str, Collection] = {}
        self.buckets: dict[str, Bucket] = {}
        
    def rejoin_any_split_chunks(self, bucket_name: str, chunks: list[Chunk]):
        return chunks
    
    def add(self, buckets: list[Bucket]):
        for bucket in buckets:
            collection = self.chroma_client.create_collection(name=bucket.name, embedding_function=self.embedder.embed)
            self.buckets[bucket.name] = bucket
            collection.add(documents=bucket.documents, embeddings=bucket.embeddings, metadatas=bucket.metadatas, ids=bucket.ids)
            self.collections[bucket.name] = collection
        
    def get(self, search_strings: list[str], n_results: int, injection_point: InjectionPoint) -> dict[InjectionPoint, list[dict]]:
        results = self.collections[injection_point.real_name].query(query_texts=search_strings, n_results=n_results)
        results = {injection_point: results}
        return results
    
    def get_ids(self, search_strings: list[str], n_results: int, injection_point: InjectionPoint, exclude_ids: list[int] = []) -> dict[InjectionPoint, list[int]]:
        where_not_in_ids = {"$and": [{"id": {"$ne": id}} for id in exclude_ids]} if len(exclude_ids) > 0 else None
        results = self.collections[injection_point.real_name].query(query_texts=search_strings, n_results=n_results, where=where_not_in_ids)['ids'][0]
        results = [int(result.split('id')[1]) for result in results]
        results = {injection_point: results}
        return results
    
    def get_chunks(self, search_strings: list[str], n_results: int, injection_point: InjectionPoint, exclude_chunks: list[Chunk] = []) -> dict[InjectionPoint, list[Chunk]]:
        where_not_in_ids = [chunk.id for chunk in exclude_chunks]
        ids = self.get_ids(search_strings, n_results, injection_point, where_not_in_ids)[injection_point]
        corresponding_bucket = self.buckets[injection_point.real_name]
        corresponding_chunks = [corresponding_bucket.chunks[id] for id in ids]
        return {injection_point: corresponding_chunks}
    
    def get_collection(self, bucket: Bucket):
        return self.collections[bucket.name]
    
    def clear(self):
        for bucket_name, bucket in self.buckets.items():
            collection = self.collections[bucket_name]
            collection.delete(bucket.ids)