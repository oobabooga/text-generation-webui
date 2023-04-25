from sentence_transformers import SentenceTransformer
import chromadb
import torch
from chromadb.api.models import Collection as ChromaCollection
from .base import Injector, Collecter, Chunker, Embedder, Retriever

class NaiveInjector(Injector):
    def __init__(self, chunker: Chunker, collector: Collecter):
        self.chunker = chunker
        self.collector = collector
        self.prepared_output = ''
    
    def prepare(self, text: str):
        all_chunks = self.chunker.make_chunks(text)
        first_chunk = all_chunks[0]
        last_chunk = all_chunks[-1]
        self.prepared_output = first_chunk + '[[[injection_point]]]' + last_chunk
        self.collector.add(all_chunks[1:-1])
        
    def inject(self, text: str) -> str:
        injected_prompt = self.prepared_output.replace('[[[injection_point]]]', text)
        print("Injected prompt: ", injected_prompt)
        return injected_prompt

class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client()
        self.embedder = embedder
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=embedder.embed)
        self.ids = []
    
    def add(self, texts: list[str]):
        self.ids = [f"id{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, ids=self.ids)
        
    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents'])['documents'][0]
        return result
    
    def clear(self):
        self.collection.delete(ids=self.ids)
    
class NaiveChunker(Chunker):
    def __init__(self, chunk_len: int, first_len: int, last_len: int):
        super().__init__(chunk_len=chunk_len, first_len=first_len, last_len=last_len)
        print(self.first_len)
        self.chunks = []
        
    def chunk(self, text: str) -> list[str]:  
        first_chunk = text[:self.first_len]
        last_chunk = text[-self.last_len:]
        middle_portion = text[self.first_len:-self.last_len]
        middle_chunks = [middle_portion[i:i + self.chunk_len] for i in range(0, len(middle_portion), self.chunk_len)]
        return [first_chunk] + middle_chunks + [last_chunk]
    
    def make_chunks(self, text: str) -> list[str]:
        self.chunks = self.chunk(text)
        return self.chunks
    
    def get_chunks(self) -> list[str]:
        return self.chunks
    
class CosineSimilarityRetriever(Retriever):
    def __init__(self, collector: Collecter, chunker: Chunker) -> None:
        self.collector = collector
        self.chunker = chunker
    
    def retrieve(self) -> list[str]:
        all_chunks = self.chunker.get_chunks()
        first_chunk = all_chunks[0] or ''
        last_chunk = all_chunks[-1] or '' if all_chunks[-1] != first_chunk else ''
        print("Searching: ", first_chunk + last_chunk)
        return self.collector.get(first_chunk + last_chunk, n_results=5)
    
class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embed = self.model.encode