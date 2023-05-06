from ..base import Source, Chunk

class MetadataChunk(Chunk):
    def add_meta(self, name: str, value: str):
        if self.meta is None:
            self.meta = {}
            
        self.meta[name] = value
    
class MetadataBuilder():
    """
    Given a source, chunk it and add metadata
    Metadata makes it easier to find chunks that are related
    """
    def enrich(self, chunk: Chunk) -> MetadataChunk:
        chunk = self.meta_id(chunk)
        return chunk
    
    def meta_related(self, chunk: Chunk) -> Chunk:
        """
        Add metadata of related chunks
        """
        pass
    
    def meta_id(self, chunk: Chunk) -> Chunk:
        """
        Add metadata of related chunks
        """
        chunk.metadatas['id'] = chunk.id
        return chunk
    
    def meta_content_type(self, chunk: Chunk) -> Chunk:
        """
        Add content type
        """
        pass
            