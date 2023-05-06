import requests
from ..base import Source

class TextSource(Source):
    def __init__(self, text=''):
        super().__init__()
        
        if len(text) > 1:
            self.set(text)
        
    def get(self) -> str:
        if self.contents is not None:
            return self.contents
        self.contents = self.text
        return super().get()
            
    def set(self, text: str):
        self.text = text
        super().set()