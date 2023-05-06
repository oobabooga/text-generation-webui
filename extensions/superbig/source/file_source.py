from ..base import Source

class FileSource(Source):
    def __init__(self):
        super().__init__()
        
    def get(self):
        if self.contents is not None:
            return self.contents
        
        with open(self.path) as f:
            self.contents = f.read()
            
        return super().get()
            
    def set(self, path: str):
        self.path = path
        super().set()