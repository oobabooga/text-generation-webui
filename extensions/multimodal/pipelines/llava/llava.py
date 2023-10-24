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


def expand2square(pil_img: Image.Image, background_color: Tuple[int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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

        logger.info(f"LLaVA - Loading CLIP from {self.CLIP_REPO} as {self.clip_dtype} on {self.clip_device}...")
        image_processor = CLIPImageProcessor.from_pretrained(self.CLIP_REPO, torch_dtype=self.clip_dtype)
        vision_tower = CLIPVisionModel.from_pretrained(self.CLIP_REPO, torch_dtype=self.clip_dtype).to(self.clip_device)

        logger.info(f"LLaVA - Loading projector from {self.llava_projector_repo()} as {self.projector_dtype} on {self.projector_device}...")
        projector_path = hf_hub_download(self.llava_projector_repo(), self.llava_projector_filename())
        mm_projector = self.build_mm_projector()
        projector_data = torch.load(projector_path)
        projector_data = {k[19:]: v for k, v in projector_data.items() if k.startswith('model.mm_projector.')}
        mm_projector.load_state_dict(projector_data)
        mm_projector = mm_projector.to(self.projector_device)

        logger.info(f"LLaVA supporting models loaded, took {time.time() - start_ts:.2f} seconds")
        return image_processor, vision_tower, mm_projector

    def build_mm_projector(self) -> torch.nn.Module:
        projector_shape = self.llava_projector_shape()
        if len(projector_shape) == 2:
            return torch.nn.Linear(*projector_shape)
        else:
            modules = []
            modules.append(torch.nn.Linear(projector_shape[0], projector_shape[1]))
            for i in range(2, len(projector_shape)):
                modules.append(torch.nn.GELU())
                modules.append(torch.nn.Linear(projector_shape[i-1], projector_shape[i]))
            return torch.nn.Sequential(*modules)

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


class LLaVA_v1_5_13B_Pipeline(LLaVA_v0_13B_Pipeline):
    CLIP_REPO = "openai/clip-vit-large-patch14-336"

    def __init__(self, params: dict) -> None:
        super().__init__(params)

    @staticmethod
    def name() -> str:
        return "llava-v1.5-13b"

    @staticmethod
    def llava_projector_shape() -> Tuple[int, int]:
        return (1024, 5120, 5120)

    @staticmethod
    def placeholder_token_id() -> int:
        return 0

    @staticmethod
    def llava_projector_repo() -> str:
        return "liuhaotian/llava-v1.5-13b"

    @staticmethod
    def image_start() -> str:
        return ""

    @staticmethod
    def image_end() -> str:
        return ""

    @staticmethod
    def num_image_embeds() -> int:
        return 576

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        # pad it to square first
        images = [
            expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            for image in images
        ]
        return super().embed_images(images)

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        return LLaVA_v0_Pipeline.embed_tokens(encode("<unk>"*576, add_bos_token=False)[0])

class LLaVA_v1_5_7B_Pipeline(LLaVA_v1_5_13B_Pipeline):
    @staticmethod
    def name() -> str:
        return "llava-v1.5-7b"

    @staticmethod
    def llava_projector_shape() -> Tuple[int, int]:
        return (1024, 4096, 4096)
    @staticmethod
    def llava_projector_repo() -> str:
        return "liuhaotian/llava-v1.5-7b"