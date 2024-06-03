import base64
from contextlib import ExitStack
from ctypes import POINTER, c_uint8, cast
import ctypes
import os
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from io import BytesIO

from modules.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.llamacpp_model import llama_cpp_lib
from modules.logging_colors import logger
from modules.text_generation import encode


class LLaVA_Cpp_Pipeline(AbstractMultimodalPipeline):
    def __init__(self, params: dict) -> None:
        try:
            if shared.args.cpu:
                import llama_cpp.llava_cpp as llava_cpp
            elif shared.args.tensorcores:
                import llama_cpp_cuda_tensorcores.llava_cpp as llava_cpp
            else:
                import llama_cpp_cuda.llava_cpp as llava_cpp
        except:
            try:
                import llama_cpp.llama_cpp as llava_cpp
            except:
                raise ImportError('LLaVA C++ library not found')
                
        super().__init__()
        self.mmproj_model_path = str(shared.llava_cpp_mmproj_path)
        self.verbose = shared.args.verbose
        self.projector_device = self._get_device("projector_device", params)
        self.projector_dtype = self._get_dtype("projector_bits", params)
        
        self._llava_cpp = llava_cpp
        self._exit_stack = ExitStack()
        self._last_image_embed: Optional[llava_cpp.CtypesPointer[llava_cpp.llava_image_embed]] = None
        self._last_image_hash: Optional[int] = None
        
        if not os.path.exists(self.mmproj_model_path):
            raise ValueError(f"Mmprojector not found at {self.mmproj_model_path}")
            
        self.clip_ctx = self._load_clip_model()
        
        def clip_free():
            with llama_cpp_lib().suppress_stdout_stderr(disable=self.verbose):
                self._llava_cpp.clip_free(self.clip_ctx)
                
        self._exit_stack.callback(clip_free)
        
        def last_image_embed_free():
            with llama_cpp_lib().suppress_stdout_stderr(disable=self.verbose):
                if self._last_image_embed is not None:
                    self._llava_cpp.llava_image_embed_free(self._last_image_embed)
                    self._last_image_embed = None
                    
                    
        self._exit_stack.callback(last_image_embed_free)

    def _load_clip_model(self):
        with llama_cpp_lib().suppress_stdout_stderr(disable=self.verbose):
            clip_ctx = self._llava_cpp.clip_model_load(
                self.mmproj_model_path.encode(), 0
            )
            
            if clip_ctx is None:
                raise ValueError(f"Failed to load Mmprojector from {self.mmproj_model_path}")
            
        return clip_ctx

    @staticmethod
    def name() -> str:
        return "Llava15ChatHandler"
    
    @staticmethod
    def image_start() -> str:
        return ""

    @staticmethod
    def image_end() -> str:
        return ""

    @staticmethod
    def placeholder_token_id() -> int:
        return 0

    @staticmethod
    def num_image_embeds() -> int:
        return 576

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert input_ids to list of token IDs
        input_ids_list = input_ids.tolist()
        
        # Step 2: Decode each token ID to corresponding tokens
        input_text = [shared.tokenizer.decode(token_id) for token_id in input_ids_list]
        
        # Step 3: Join tokens to form the input string
        input_string = ''.join(input_text)

        # Step 4: Use create_embedding method from llama_cpp to get the embedding
        embedding_response = shared.model.model.create_embedding(input=input_string)
        
        # Step 5: Extract embedding data and check dimensions
        embeddings = []
        for embed in embedding_response["data"]:
            embedding_tensor = torch.tensor(embed["embedding"]).to(self.projector_device)
            embeddings.append(embedding_tensor)
    
        # Step 6: Stitch embeddings into a single (1*n) tensor
        embeddings_tensor = torch.cat(embeddings, dim=0)
        print(f"Embeddings tensor shape: {embeddings_tensor.shape}")
        
        return embeddings_tensor

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return LLaVA_Cpp_Pipeline.embed_tokens(encode("<unk>"*576, add_bos_token=False)[0])

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        image_embeddings = []

        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_bytes_length = len(image_bytes)
            image_bytes_array = (ctypes.c_uint8 * image_bytes_length).from_buffer_copy(image_bytes)
            image_embed_ptr = self._llava_cpp.llava_image_embed_make_with_bytes(
                self.clip_ctx,
                shared.args.threads_batch,
                image_bytes_array,
                image_bytes_length
            )

            if not image_embed_ptr:
                raise ValueError("llava_image_embed_make_with_bytes returned NULL pointer")

            image_embed = image_embed_ptr.contents
            embed_array = np.ctypeslib.as_array(image_embed.embed, (image_embed.n_image_pos, shared.model.model.n_embd()))
            image_embedding = torch.tensor(embed_array)
            image_embeddings.append(image_embedding)
            print(f"Image embedding shape: {image_embedding.shape}")


        image_embeddings_tensor = torch.stack(image_embeddings)
        print(f"Image embeddings tensor shape: {image_embeddings_tensor.shape}")
        image_embeddings_tensor = image_embeddings_tensor.to(self.projector_device, dtype=self.projector_dtype)
        return image_embeddings_tensor
