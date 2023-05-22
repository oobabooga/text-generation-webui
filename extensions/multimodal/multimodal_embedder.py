import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional

import torch
from PIL import Image

from extensions.multimodal.pipeline_loader import load_pipeline
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import encode, get_max_prompt_length


@dataclass
class PromptPart:
    text: str
    image: Optional[Image.Image] = None
    is_image: bool = False
    input_ids: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class MultimodalEmbedder:
    def __init__(self, params: dict):
        pipeline, source = load_pipeline(params)
        self.pipeline = pipeline
        logger.info(f'Multimodal: loaded pipeline {self.pipeline.name()} from pipelines/{source} ({self.pipeline.__class__.__name__})')

    def _split_prompt(self, prompt: str, load_images: bool = False) -> List[PromptPart]:
        """Splits a prompt into a list of `PromptParts` to separate image data from text.
        It will also append `image_start` and `image_end` before and after the image, and optionally parse and load the images,
        if `load_images` is `True`.
        """
        parts: List[PromptPart] = []
        curr = 0
        while True:
            match = re.search(r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)">', prompt[curr:])
            if match is None:
                # no more image tokens, append the rest of the prompt
                if curr > 0:
                    # add image end token after last image
                    parts.append(PromptPart(text=self.pipeline.image_end() + prompt[curr:]))
                else:
                    parts.append(PromptPart(text=prompt))
                break
            # found an image, append image start token to the text
            if match.start() > 0:
                parts.append(PromptPart(text=prompt[curr:curr + match.start()] + self.pipeline.image_start()))
            else:
                parts.append(PromptPart(text=self.pipeline.image_start()))
            # append the image
            parts.append(PromptPart(
                text=match.group(0),
                image=Image.open(BytesIO(base64.b64decode(match.group(1)))) if load_images else None,
                is_image=True
            ))
            curr += match.end()
        return parts

    def _len_in_tokens_prompt_parts(self, parts: List[PromptPart]) -> int:
        """Total length in tokens of all `parts`"""
        tokens = 0
        for part in parts:
            if part.is_image:
                tokens += self.pipeline.num_image_embeds()
            elif part.input_ids is not None:
                tokens += len(part.input_ids)
            else:
                tokens += len(encode(part.text)[0])
        return tokens

    def len_in_tokens(self, prompt: str) -> int:
        """Total length in tokens for a given text `prompt`"""
        parts = self._split_prompt(prompt, False)
        return self._len_in_tokens_prompt_parts(parts)

    def _encode_single_text(self, part: PromptPart, add_bos_token: bool) -> PromptPart:
        """Encode a single prompt `part` to `input_ids`. Returns a `PromptPart`"""
        if part.is_image:
            placeholders = torch.ones((self.pipeline.num_image_embeds())) * self.pipeline.placeholder_token_id()
            part.input_ids = placeholders.to(shared.model.device, dtype=torch.int64)
        else:
            part.input_ids = encode(part.text, add_bos_token=add_bos_token)[0].to(shared.model.device, dtype=torch.int64)
        return part

    @staticmethod
    def _num_images(parts: List[PromptPart]) -> int:
        count = 0
        for part in parts:
            if part.is_image:
                count += 1
        return count

    def _encode_text(self, state, parts: List[PromptPart]) -> List[PromptPart]:
        """Encode text to token_ids, also truncate the prompt, if necessary.

        The chat/instruct mode should make prompts that fit in get_max_prompt_length, but if max_new_tokens are set
        such that the context + min_rows don't fit, we can get a prompt which is too long.
        We can't truncate image embeddings, as it leads to broken generation, so remove the images instead and warn the user
        """
        encoded: List[PromptPart] = []
        for i, part in enumerate(parts):
            encoded.append(self._encode_single_text(part, i == 0 and state['add_bos_token']))

        # truncation:
        max_len = get_max_prompt_length(state)
        removed_images = 0

        # 1. remove entire text/image blocks
        while self._len_in_tokens_prompt_parts(encoded[1:]) > max_len:
            if encoded[0].is_image:
                removed_images += 1
            encoded = encoded[1:]

        # 2. check if the last prompt part doesn't need to get truncated
        if self._len_in_tokens_prompt_parts(encoded) > max_len:
            if encoded[0].is_image:
                # don't truncate image embeddings, just remove the image, otherwise generation will be broken
                removed_images += 1
                encoded = encoded[1:]
            elif len(encoded) > 1 and encoded[0].text.endswith(self.pipeline.image_start()):
                # see if we can keep image_start token
                len_image_start = len(encode(self.pipeline.image_start(), add_bos_token=state['add_bos_token'])[0])
                if self._len_in_tokens_prompt_parts(encoded[1:]) + len_image_start > max_len:
                    # we can't -> remove this text, and the image
                    encoded = encoded[2:]
                    removed_images += 1
                else:
                    # we can -> just truncate the text
                    trunc_len = self._len_in_tokens_prompt_parts(encoded) - max_len
                    encoded[0].input_ids = encoded[0].input_ids[trunc_len:]
            elif len(encoded) > 0:
                # only one text left, truncate it normally
                trunc_len = self._len_in_tokens_prompt_parts(encoded) - max_len
                encoded[0].input_ids = encoded[0].input_ids[trunc_len:]

        # notify user if we truncated an image
        if removed_images > 0:
            logger.warning(f"Multimodal: removed {removed_images} image(s) from prompt. Try decreasing max_new_tokens if generation is broken")

        return encoded

    def _embed(self, parts: List[PromptPart]) -> List[PromptPart]:
        # batch images
        image_indicies = [i for i, part in enumerate(parts) if part.is_image]
        embedded = self.pipeline.embed_images([parts[i].image for i in image_indicies])
        for i, embeds in zip(image_indicies, embedded):
            parts[i].embedding = embeds
        # embed text
        for (i, part) in enumerate(parts):
            if not part.is_image:
                parts[i].embedding = self.pipeline.embed_tokens(part.input_ids)
        return parts

    def _remove_old_images(self, parts: List[PromptPart], params: dict) -> List[PromptPart]:
        if params['add_all_images_to_prompt']:
            return parts
        already_added = False
        for i, part in reversed(list(enumerate(parts))):
            if part.is_image:
                if already_added:
                    parts[i].embedding = self.pipeline.placeholder_embeddings()
                else:
                    already_added = True
        return parts

    def forward(self, prompt: str, state: Any, params: dict):
        prompt_parts = self._split_prompt(prompt, True)
        prompt_parts = self._encode_text(state, prompt_parts)
        prompt_parts = self._embed(prompt_parts)
        prompt_parts = self._remove_old_images(prompt_parts, params)
        embeds = tuple(part.embedding for part in prompt_parts)
        ids = tuple(part.input_ids for part in prompt_parts)
        input_embeds = torch.cat(embeds, dim=0)
        input_ids = torch.cat(ids, dim=0)
        return prompt, input_ids, input_embeds, self._num_images(prompt_parts)
