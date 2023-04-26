from ..base import Chunker, Collecter, Injector


class InstructInjector(Injector):
    def __init__(self, chunker: Chunker, collector: Collecter):
        self.chunker = chunker
        self.collector = collector
        self.prepared_output = ''
    
    def prepare(self, text: str):
        all_chunks = self.chunker.make_chunks(text)
        instruct_chunk = all_chunks[0]
        data_chunks = [element for i, element in enumerate(all_chunks) if i not in (0, len(all_chunks) - 2)]
        input_chunk = all_chunks[-2]
        response_chunk = all_chunks[-1]
        self.prepared_output = instruct_chunk + '\n\n[[[injection_point]]]\n\n' + input_chunk + response_chunk
        self.collector.add(data_chunks)
        print("Template:\n", self.prepared_output)
        
    def inject(self, text: str) -> str:
        injected_prompt = self.prepared_output.replace('[[[injection_point]]]', text)
        print("Injected prompt: ", injected_prompt)
        return injected_prompt