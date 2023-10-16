import time
from abc import abstractmethod
from typing import List, Tuple

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode


class LLaVA_v0_Pipeline(AbstractMultimodalPipeline):
    CLIP_REPO = "openai/clip-vit-large-patch14"

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.clip_device = self._get_device("vision_device", params)
        self.clip_dtype = self._get_dtype("vision_bits", params)
        self.projector_device = self._get_device("projector_device", params)
        self.projector_dtype = self._get_dtype("projector_bits", params)
        self.image_processor, self.vision_tower, self.mm_projector = self._load_models()

    def _load_models(self):
        start_ts = time.time()

        logger.info(f"LLaVA - Loading CLIP from {LLaVA_v0_Pipeline.CLIP_REPO} as {self.clip_dtype} on {self.clip_device}...")
        image_processor = CLIPImageProcessor.from_pretrained(LLaVA_v0_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype)
        vision_tower = CLIPVisionModel.from_pretrained(LLaVA_v0_Pipeline.CLIP_REPO, torch_dtype=self.clip_dtype).to(self.clip_device)

        logger.info(f"LLaVA - Loading projector from {self.llava_projector_repo()} as {self.projector_dtype} on {self.projector_device}...")
        projector_path = hf_hub_download(self.llava_projector_repo(), self.llava_projector_filename())
        mm_projector = torch.nn.Linear(*self.llava_projector_shape())
        projector_data = torch.load(projector_path)
        mm_projector.weight = torch.nn.Parameter(projector_data['model.mm_projector.weight'].to(dtype=self.projector_dtype), False)
        mm_projector.bias = torch.nn.Parameter(projector_data['model.mm_projector.bias'].to(dtype=self.projector_dtype), False)
        mm_projector = mm_projector.to(self.projector_device)

        logger.info(f"LLaVA supporting models loaded, took {time.time() - start_ts:.2f} seconds")
        return image_processor, vision_tower, mm_projector

    @staticmethod
    def image_start() -> str:
        return "<im_start>"

    @staticmethod
    def image_end() -> str:
        return "<im_end>"

    @staticmethod
    def num_image_embeds() -> int:
        return 256

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        for attr in ['', 'model', 'model.model', 'model.model.model']:
            tmp = getattr(shared.model, attr, None) if attr != '' else shared.model
            if tmp is not None and hasattr(tmp, 'embed_tokens'):
                func = tmp.embed_tokens
                break
        else:
            raise ValueError('The embed_tokens method has not been found for this loader.')

        return func(input_ids).to(shared.model.device, dtype=shared.model.dtype)

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return LLaVA_v0_Pipeline.embed_tokens(encode("<im_patch>"*256, add_bos_token=False)[0])

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        images = self.image_processor(images, return_tensors='pt')['pixel_values']
        images = images.to(self.clip_device, dtype=self.clip_dtype)

        with torch.no_grad():
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            select_hidden_state_layer = -2
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:].to(self.projector_device, dtype=self.projector_dtype)
            image_features = self.mm_projector(image_features)
        return image_features.to(shared.model.device, dtype=shared.model.dtype)

    @staticmethod
    @abstractmethod
    def llava_projector_repo() -> str:
        pass

    @staticmethod
    @abstractmethod
    def llava_projector_filename() -> str:
        pass

    @staticmethod
    @abstractmethod
    def llava_projector_shape() -> Tuple[int, int]:
        pass


class LLaVA_v0_13B_Pipeline(LLaVA_v0_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "llava-13b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 32000

    @staticmethod
    def llava_projector_shape() -> Tuple[int, int]:
        return (1024, 5120)

    @staticmethod
    def llava_projector_filename() -> str:
        return "mm_projector.bin"

    @staticmethod
    def llava_projector_repo() -> str:
        return "liuhaotian/LLaVA-13b-delta-v0"


class LLaVA_v0_7B_Pipeline(LLaVA_v0_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "llava-7b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 32001

    @staticmethod
    def llava_projector_shape() -> Tuple[int, int]:
        return (1024, 4096)

    @staticmethod
    def llava_projector_filename() -> str:
        return "mm_projector.bin"

    @staticmethod
    def llava_projector_repo() -> str:
        return "liuhaotian/LLaVA-7b-delta-v0"


class LLaVA_LLaMA_2_13B_Pipeline(LLaVA_v0_13B_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "llava-llama-2-13b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 0

    @staticmethod
    def llava_projector_repo() -> str:
        return "liuhaotian/llava-llama-2-13b-chat-lightning-preview"

    @staticmethod
    def image_start() -> str:
        return ""

    @staticmethod
    def image_end() -> str:
        return ""

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return LLaVA_v0_Pipeline.embed_tokens(encode("<unk>"*256, add_bos_token=False)[0])
