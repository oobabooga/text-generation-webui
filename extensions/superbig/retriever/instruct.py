from ..base import Collecter, Chunker, Retriever

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