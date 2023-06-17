import sys
from pathlib import Path

from modules import shared
from modules.logging_colors import logger

sys.path.insert(0, str(Path("repositories/exllama")))
from repositories.exllama.generator import ExLlamaGenerator
from repositories.exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from repositories.exllama.tokenizer import ExLlamaTokenizer


class ExllamaModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path("models") / Path(path_to_model)
        tokenizer_model_path = path_to_model / "tokenizer.model"
        model_config_path = path_to_model / "config.json"

        # Find the model checkpoint
        model_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                model_path = found[-1]
                break

        config = ExLlamaConfig(str(model_config_path))
        config.model_path = str(model_path)
        if shared.args.gpu_split:
            config.set_auto_map(shared.args.gpu_split)
            config.gpu_peer_fix = True

        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
        cache = ExLlamaCache(model)

        result = self()
        result.config = config
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        return result, result

    def generate(self, prompt, state, callback=None):
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings.temperature = state['temperature']
        generator.settings.top_p = state['top_p']
        generator.settings.top_k = state['top_k']
        generator.settings.typical = state['typical_p']
        generator.settings.token_repetition_penalty_max = state['repetition_penalty']
        if state['ban_eos_token']:
            generator.disallow_tokens([self.tokenizer.eos_token_id])

        text = generator.generate_simple(prompt, max_new_tokens=state['max_new_tokens'])
        return text

    def generate_with_streaming(self, prompt, state, callback=None):
        generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        generator.settings.temperature = state['temperature']
        generator.settings.top_p = state['top_p']
        generator.settings.top_k = state['top_k']
        generator.settings.typical = state['typical_p']
        generator.settings.token_repetition_penalty_max = state['repetition_penalty']
        if state['ban_eos_token']:
            generator.disallow_tokens([self.tokenizer.eos_token_id])

        generator.end_beam_search()
        ids = generator.tokenizer.encode(prompt)
        generator.gen_begin(ids)
        initial_len = generator.sequence[0].shape[0]
        for i in range(state['max_new_tokens']):
            token = generator.gen_single_token()
            yield (generator.tokenizer.decode(generator.sequence[0][initial_len:]))
            if token.item() == generator.tokenizer.eos_token_id or shared.stop_everything:
                break

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string)
