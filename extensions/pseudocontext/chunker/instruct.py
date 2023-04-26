from ..base import Chunker


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