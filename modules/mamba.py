import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
from modules.logging_colors import logger
from modules.callbacks import Iteratorize


class MambaSsmModel:

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        model = MambaLMHeadModel.from_pretrained(path_to_model, dtype=torch.bfloat16, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

        result = self()
        result.model = model
        result.cache = None
        result.tokenizer = tokenizer
        result.generator = None
        result.loras = None
        return result, tokenizer

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, return_tensors='pt')

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, decode_special_tokens=True)

    def get_logits(self, token_ids, **kwargs):
        logger.debug("mamba: token_ids %s", token_ids)
        output = self.model.generate(
            input_ids=token_ids.cuda(),
            max_length=8,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
        )
        logger.debug("mamba: output %s", output)
        logger.debug("mamba: output.scores %s", output.scores)
        scores = output.scores[0]
        logger.debug("scores %s", scores)
        logger.debug("scores[-1] %s", scores[-1])
        logger.debug("scores[-1][-1] %s", scores[-1][-1])

        raise NotImplementedError("logit results look wrong right now")
        return scores

    def generate(self, prompt, state, callback=None):
        input_ids = self.encode(prompt)
        initial_len = len(input_ids[0])

        output = self.model.generate(
            input_ids=input_ids.cuda(),
            max_length=state['max_new_tokens'] + initial_len,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=state['temperature'],
            top_k=state['top_k'],
            top_p=state['top_p'],
        )
        decoded = self.decode(output.sequences.cpu()[0][initial_len:])
        logger.debug("decoded %s", decoded)
        callback(decoded)
        return decoded

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs,
                         callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)