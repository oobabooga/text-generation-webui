from ..base import Chunker, Embedder, Collecter, Retriever, Injector
from ..chunker import NaiveChunker, InstructChunker
from ..embedder import SentenceTransformerEmbedder
from ..collector import ChromaCollector
from ..retriever import CosineSimilarityRetriever, InstructRetriever
from ..injector import NaiveInjector, InstructInjector

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
        self.retriever = retriever or CosineSimilarityRetriever(self.collector, self.chunker)
        self.injector = injector or NaiveInjector(self.chunker, self.collector)
        
    def __enter__(self):
        return self.with_pseudocontext(self.prompt)
    
    def __exit__(self, type, value, trace):
        pass
        
    def with_pseudocontext(self, prompt: str):
        self.injector.prepare(prompt)
        relevant_context = self.retriever.retrieve()
        self.collector.clear()
        return self.injector.inject("\n".join(relevant_context))
    
def make_instruct_provider():
    chunker = InstructChunker(chunk_len=700)
    embedder = SentenceTransformerEmbedder()
    collector = ChromaCollector(embedder)
    retriever = InstructRetriever(collector, chunker)
    injector = InstructInjector(chunker, collector)
    instruct_provider = PseudocontextProvider(chunker=chunker, embedder=embedder, collector=collector, retriever=retriever, injector=injector)
    return instruct_provider