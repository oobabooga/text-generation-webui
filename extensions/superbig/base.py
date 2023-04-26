import torch

class Collecter():
    def __init__(self):
        pass
    
    def add(self, texts: list[str]):
        pass
    
    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        pass
    
    def clear(self):
        pass
    
class Chunker():
    def __init__(self, chunk_len: int, first_len: int, last_len: int):
        self.chunk_len = chunk_len
        self.first_len = first_len
        self.last_len = last_len
        
    def chunk(self, text: str) -> list[str]:
        pass
    
    def make_chunks(self, text: str) -> list[str]:
        pass
    
    def get_chunks(self) -> list[str]:
        pass

class Retriever():
    def __init__(self):
        pass
    
    def retrieve(self) -> list[str]:
        pass

class Embedder():
    def __init__(self):
        pass
    
    def embed(self, text: str) -> list[torch.Tensor]:
        pass
    
class Injector():
    def __init__(self, chunker: Chunker, collector: Collecter):
        pass
    
    def prepare(self, text: str):
        pass 
    
    def inject(self, text: str) -> str:
        pass
    
class Focuser():
    def __init__(self):
        pass
    
    def focus(texts: list[str]) -> list[str]:
        pass