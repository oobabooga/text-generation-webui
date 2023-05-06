from ..base import Chunker, Embedder, Collecter, Retriever, Injector
from ..chunker import NaiveChunker
from ..embedder import SentenceTransformerEmbedder
from ..collector import ChromaCollector
from ..injector import GenericInjector

class PseudocontextProvider():
    def __init__(self,
                 prompt: str = '',
                 collector: Collecter = None, 
                 chunker: Chunker = None, 
                 retriever: Retriever = None, 
                 embedder: Embedder = None, 
                 injector: Injector = None,
                 chunk_len: int = 500, 
                 first_len: int = 300, 
                 last_len:int = 300):
        self.prompt = prompt
        self.chunker = chunker or NaiveChunker(chunk_len=chunk_len, first_len=first_len, last_len=last_len)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.collector = collector or ChromaCollector(self.embedder)
        self.injector = injector or GenericInjector(self.chunker, self.collector, self.embedder, {})
        
    def __enter__(self):
        return self.with_pseudocontext(self.prompt)
    
    def __exit__(self, type, value, trace):
        pass
        
    def with_pseudocontext(self, prompt: str):
        prepared_prompt = self.injector.prepare(prompt)
        new_prompt = self.injector.inject(prepared_prompt)
        return new_prompt