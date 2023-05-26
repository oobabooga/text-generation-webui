from ctransformers import AutoModelForCausalLM
from ctransformers import AutoConfig

from modules import shared


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
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), model_type="starcoder", config=config
        )
        return result, result

    def encode(self, string, **kwargs):
        return self.model.tokenize(string)

    def decode(self, ids):
        return self.model.detokenize(ids)

    def generate(
        self,
        context="",
        token_count=20,
        temperature=1,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        callback=None,
    ):
        context = context if type(context) is str else context.decode()
        generator = self.model._stream(
            prompt=context,
            max_new_tokens=token_count,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            threads=shared.args.threads,
        )
        out = ""
        for token in generator:
            if callback:
                callback(token)
            out += token
        return out

    def generate_with_streaming(
        self,
        context="",
        token_count=20,
        temperature=1,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        callback=None,
    ):
        context = context if type(context) is str else context.decode()
        generator = self.model._stream(
            prompt=context,
            max_new_tokens=token_count,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            threads=shared.args.threads,
        )
        reply = ""
        for token in generator:
            reply += token
            yield reply
