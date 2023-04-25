from sentence_transformers import SentenceTransformer
import chromadb
import torch
from chromadb.api.models import Collection as ChromaCollection

from extensions.pseudocontext.provider import PseudocontextProvider
from extensions.pseudocontext.test_models import ChromaCollector, SentenceTransformerEmbedder
from .base import Injector, Collecter, Chunker, Embedder, Retriever

class InstructInjector(Injector):
    def __init__(self, chunker: Chunker, collector: Collecter):
        self.chunker = chunker
        self.collector = collector
        self.prepared_output = ''
    
    def prepare(self, text: str):
        all_chunks = self.chunker.make_chunks(text)
        instruct_chunk = all_chunks[0]
        data_chunks = [element for i, element in enumerate(all_chunks) if i not in (0, len(all_chunks) - 2)]
        input_chunk = all_chunks[-2]
        response_chunk = all_chunks[-1]
        self.prepared_output = instruct_chunk + '\n\n[[[injection_point]]]\n\n' + input_chunk + response_chunk
        self.collector.add(data_chunks)
        print("Template:\n", self.prepared_output)
        
    def inject(self, text: str) -> str:
        injected_prompt = self.prepared_output.replace('[[[injection_point]]]', text)
        print("Injected prompt: ", injected_prompt)
        return injected_prompt
    
class InstructChunker(Chunker):
    def __init__(self, chunk_len: int, first_len: int = 0, last_len: int = 0):
        super().__init__(chunk_len=chunk_len, first_len=first_len, last_len=last_len)
        self.chunks = []
        
    def chunk(self, text: str) -> list[str]:
        instruct_idx = 0
        data_idx = text.index("### Data:")
        input_idx = text.index("### Input:")
        response_idx = text.index("### Response:")
        
        instruct_chunk = text[instruct_idx:data_idx].strip()
        data_portion = text[data_idx:input_idx]
        input_chunk = text[input_idx:response_idx]
        response_chunk = text[response_idx:]
        data_chunks = [data_portion[i:i + self.chunk_len] for i in range(0, len(data_portion), self.chunk_len)]
        
        return [instruct_chunk] + data_chunks + [input_chunk, response_chunk]
    
    def make_chunks(self, text: str) -> list[str]:
        self.chunks = self.chunk(text)
        return self.chunks
    
    def get_chunks(self) -> list[str]:
        return self.chunks
    
class InstructRetriever(Retriever):
    def __init__(self, collector: Collecter, chunker: Chunker) -> None:
        self.collector = collector
        self.chunker = chunker
    
    def retrieve(self) -> list[str]:
        all_chunks = self.chunker.get_chunks()
        input_chunk = all_chunks[-2]
        input_chunk = input_chunk.replace("### Input:", '')
        print("Searching: ", input_chunk)
        return self.collector.get(input_chunk, n_results=5)
    
def make_instruct_provider():
    chunker = InstructChunker(chunk_len=1000)
    embedder = SentenceTransformerEmbedder()
    collector = ChromaCollector(embedder)
    retriever = InstructRetriever(collector, chunker)
    injector = InstructInjector(chunker, collector)
    instruct_provider = PseudocontextProvider(chunker=chunker, embedder=embedder, collector=collector, retriever=retriever, injector=injector)
    return instruct_provider