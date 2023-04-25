from .base import Chunker, Embedder, Collecter, Retriever, Injector
from .test_models import NaiveChunker, SentenceTransformerEmbedder, ChromaCollector, CosineSimilarityRetriever, NaiveInjector

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
        
    # proof of concept, this will be replaced with more sophisticated chunker/retrievers and selecting the appropriate one based on settings
    def with_pseudocontext(self, prompt: str):
        # chunk the context and prepare it for injection and add to the store
        self.injector.prepare(prompt)
        # query the chunks to get relevant portions of the context
        relevant_context = self.retriever.retrieve()
        print(relevant_context)
        # clear the db for the next run
        self.collector.clear()
        # inject the relevant context in the appropriate spot
        return self.injector.inject("\n".join(relevant_context))