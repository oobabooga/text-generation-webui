from ..base import Collecter, Chunker, Retriever

class CosineSimilarityRetriever(Retriever):
    def __init__(self, collector: Collecter, chunker: Chunker) -> None:
        self.collector = collector
        self.chunker = chunker
    
    def retrieve(self) -> list[str]:
        all_chunks = self.chunker.get_chunks()
        first_chunk = all_chunks[0] or ''
        last_chunk = all_chunks[-1] or '' if all_chunks[-1] != first_chunk else ''
        print("Searching string: ", first_chunk + last_chunk)
        return self.collector.get(first_chunk + last_chunk, n_results=5)