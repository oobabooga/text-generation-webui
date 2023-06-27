from ctransformers import AutoModelForCausalLM
from ctransformers import AutoConfig

from modules import shared
from modules.callbacks import Iteratorize


class StarcoderCppModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        result = self()

        config = AutoConfig.from_pretrained(
            str(path),
            stop=["<|end|>"],
            threads=shared.args.threads,
            gpu_layers=shared.args.n_gpu_layers
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), model_type="starcoder", config=config
        )
        return result, result

    def encode(self, string, **kwargs):
        return self.model.tokenize(string)

    def decode(self, ids):
        return self.model.detokenize(ids)


    def generate(self, prompt, state, callback=None):
        #try:
        prompt = prompt if type(prompt) is str else prompt.decode()
        generator = self.model._stream(
            prompt=prompt,
            max_new_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            repetition_penalty=state['repetition_penalty'],
            threads=shared.args.threads
        )

        output = ""
        for token in generator:
            if callback:
                callback(token)
            output += token
        return output


    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
