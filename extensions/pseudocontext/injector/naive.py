from ..base import Collecter, Chunker, Injector

class NaiveInjector(Injector):
    def __init__(self, chunker: Chunker, collector: Collecter):
        self.chunker = chunker
        self.collector = collector
        self.prepared_output = ''
    
    def prepare(self, text: str):
        all_chunks = self.chunker.make_chunks(text)
        first_chunk = all_chunks[0]
        last_chunk = all_chunks[-1]
        self.prepared_output = first_chunk + '[[[injection_point]]]' + last_chunk
        self.collector.add(all_chunks[1:-1])
        
    def inject(self, text: str) -> str:
        injected_prompt = self.prepared_output.replace('[[[injection_point]]]', text)
        print("Injected prompt: ", injected_prompt)
        return injected_prompt