from ..base import Embedder, Source

class SourceSelector():
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
    
    def find_best_source(self, sources: list[Source], input: str):
        description_embeddings = [self.embedder.embed(source.metadata.attributes.get('description')) for source in sources]
        input_embeddings = self.embedder.embed(input)