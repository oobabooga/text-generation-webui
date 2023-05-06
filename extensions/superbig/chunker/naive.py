from ..base import Chunker, Chunk

class NaiveChunker(Chunker):
    def __init__(self, chunk_len: int, first_len: int, last_len: int):
        super().__init__(chunk_len=chunk_len, first_len=first_len, last_len=last_len)
        self.chunks = []
        
    def chunk(self, text: str) -> list[Chunk]:  
        first_chunk = text[:self.first_len]
        last_chunk = text[-self.last_len:]
        middle_portion = text[self.first_len:-self.last_len]
        middle_chunks = [Chunk(middle_portion[i:i + self.chunk_len]) for i in range(0, len(middle_portion), self.chunk_len)]
        return [Chunk(first_chunk)] + middle_chunks + [Chunk(last_chunk)]
    
    def make_chunks(self, text: str) -> list[str]:
        self.chunks = self.chunk(text)
        return self.chunks
    
    def get_chunks(self) -> list[str]:
        return self.chunks