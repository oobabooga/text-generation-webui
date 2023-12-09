
import contextlib
from pathlib import Path
import argparse
import threading

from typing import List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.utils import random_uuid

from modules import shared
from modules.logging_colors import logger

# Lock vllm to prevent multiple threads from using it at the same time
class LockContextManager:
    def __init__(self, lock):
        self.lock = lock
        
    def __enter__(self):
        self.lock.acquire()
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.lock.release()

class VllmModel:
    __VLLM_DEBUG__ = False
    def __init__(self):
        self.inference_lock = threading.Lock()
        pass

    @classmethod
    def from_pretrained(self, path_to_model):
        
        # Parse the arguments, but ignore textgen arguments, only parse vllm arguments
        vllm_parser = argparse.ArgumentParser(
            description='VllmModel underlyingly uses the Vllm LLMEngine class directly, we will use Vllm argparser to parse the arguments instead')
        vllm_parser = EngineArgs.add_cli_args(vllm_parser)
        vllm_args, unknown = vllm_parser.parse_known_args()
        
        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)
        assert path_to_model.exists(), f'Model {path_to_model} does not exist'
        
        vllm_args.model = str(path_to_model.absolute())
        
        logger.info(f'Parsed vllm_args : {vllm_args}')
        engine_args = EngineArgs.from_cli_args(vllm_args)
        engine = LLMEngine.from_engine_args(engine_args)
        
        result = self()
        result.engine = engine
        result.tokenizer = engine.tokenizer
        
        logger.info(f'Loaded model into \n{result.engine}, \n{result.tokenizer}')
        
        return result, result.tokenizer


    def generate_with_streaming(self, prompt, state):
        
        # Get sampling settings from textgen
        settings = SamplingParams()
        for key, value in state.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
                if shared.args.verbose and self.__VLLM_DEBUG__:
                    logger.debug(f'Setting {key} to {value}')

        # use Vllm to Verify the settings
        try:
            settings._verify_args()
        except ValueError as e:
            settings = SamplingParams()
            logger.warning(f'Vllm Error verifying settings, useing default sampler settings instead {settings}: {e}')
        
        # Get prompt token prompt_token_ids
        prompt_token_ids = self.tokenizer.encode(prompt)
        # Get max new tokens
        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - len(prompt_token_ids)
        else:
            max_new_tokens = state['max_new_tokens']
        if max_new_tokens < 0:
            logger.warning(f'Max new tokens {max_new_tokens} < 0, setting to 0')
            max_new_tokens = 0
        settings.max_tokens = max_new_tokens
        
        if shared.args.verbose and self.__VLLM_DEBUG__:
            logger.debug(f'Generating with streaming, max_tokens={settings.max_tokens}')
            logger.debug(f'Prompt token ids {prompt_token_ids}')
            logger.debug(f'Prompt token ids length {len(prompt_token_ids)}')
            logger.debug(f'settings {settings}')
            
        # Can only handle 1 sample per generation
        assert settings.n == 1, f'Only 1 sample per generation is supported, got {settings.n}'
        
        request_id = f"{random_uuid()}"
        with LockContextManager(self.inference_lock):
            self.engine.add_request(request_id=request_id, 
                                prompt=prompt, 
                                sampling_params=settings, 
                                prompt_token_ids=prompt_token_ids)

        while True:
            # Abort generation if we are stopping everything
            if shared.stop_everything:
                with LockContextManager(self.inference_lock):
                    self.engine.abort(request_id)
                if shared.args.verbose and self.__VLLM_DEBUG__:
                    logger.debug(f'Aborted generation')
                break
            
            target_request_output = None
            with LockContextManager(self.inference_lock):
                request_outputs: List[RequestOutput] = self.engine.step()
            
            for request_output in request_outputs:
                if request_output.request_id != request_id:
                    logger.warning(f'Request id mismatch, expected {request_id}, got {request_output.request_id}')
                    continue
                # Can only handle 1 sample per generation
                assert len(request_output.outputs) == 1, f'Only 1 sample per generation is supported, got {len(request_output.outputs)}'
                target_request_output = request_output
                
            output = target_request_output.outputs[0]
            decoded_text = output.text
            # if shared.args.verbose and self.__VLLM_DEBUG__:
            #     logger.debug(f'{decoded_text}')
            yield decoded_text
            
            if target_request_output.finished:
                if shared.args.verbose and self.__VLLM_DEBUG__:
                    logger.debug(f'Finished generation')
                break
        

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output
