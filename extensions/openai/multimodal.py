"""
Utilities for handling multimodal input.
"""
import base64
import io
from typing import List, Tuple, Any
from PIL import Image

from modules import shared
from modules.logging_colors import logger


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decodes a base64 string to a PIL Image."""
    try:
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',', 1)[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")


def process_message_content(content: Any) -> Tuple[str, List[Image.Image]]:
    """
    Processes message content that may contain text and images.

    Returns:
        A tuple of (text_content, list_of_pil_images).
    """
    if isinstance(content, str):
        return content, []

    if isinstance(content, list):
        text_parts = []
        images = []
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get('type', '')
            if item_type == 'text':
                text_parts.append(item.get('text', ''))
            elif item_type == 'image_url':
                image_url_data = item.get('image_url', {})
                image_url = image_url_data.get('url', '')

                if image_url.startswith('data:image/'):
                    try:
                        images.append(decode_base64_image(image_url))
                    except Exception as e:
                        logger.warning(f"Failed to process a base64 image: {e}")
                elif image_url.startswith('http'):
                    # Support external URLs
                    try:
                        import requests
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        image_data = response.content
                        image = Image.open(io.BytesIO(image_data))
                        images.append(image)
                        logger.info(f"Successfully loaded external image from URL")
                    except Exception as e:
                        logger.warning(f"Failed to fetch external image: {e}")
                else:
                    logger.warning(f"Unsupported image URL format: {image_url[:70]}...")

        return ' '.join(text_parts), images

    return str(content), []


def process_unified_images(body: dict) -> List[Image.Image]:
    """
    Unified image processor for both /chat/completions and /completions endpoints.
    Always expects OpenAI image_url format, handles both endpoints consistently.
    
    Returns:
        List of PIL Images ready for multimodal processing.
    """
    raw_images = []
    
    if not is_multimodal_model():
        return raw_images
    
    try:
        # Handle /chat/completions format (messages array)
        if 'messages' in body:
            for message in body.get('messages', []):
                if isinstance(message, dict) and 'content' in message:
                    _, images = process_message_content(message['content'])
                    raw_images.extend(images)
        
        # Handle /completions format - check if multimodal content exists
        # Look for any image-related parameters and convert to messages format
        elif 'prompt' in body:
            # Check if there are any image parameters that need processing
            # This would be where we'd handle future image parameters for completions
            # For now, /completions should use messages format internally for images
            pass
            
    except Exception as e:
        logger.error(f"Error processing images in unified processor: {e}")
    
    return raw_images


def is_multimodal_model() -> bool:
    """Checks if the currently loaded model is a multimodal ExLlamaV3 model."""
    try:
        model_class_name = shared.model.__class__.__name__
        if 'Exllamav3' in model_class_name:
            return hasattr(shared.model, 'vision_model') and shared.model.vision_model is not None
        return False
    except Exception as e:
        logger.error(f"Error checking for multimodal model: {e}")
        return False


def get_image_embeddings(images: List[Image.Image]) -> List[Any]:
    """Generates image embeddings using the ExLlamaV3 model."""
    if not images or not is_multimodal_model():
        return []

    try:
        if not hasattr(shared.model, 'vision_model') or shared.model.vision_model is None:
            logger.error("ExLlamaV3 vision model is not available for embedding generation.")
            return []

        vision_model = shared.model.vision_model
        tokenizer = shared.model.tokenizer

        # Do not reset the cache/allocator index; it causes token ID conflicts during generation.

        logger.info(f"Processing {len(images)} image(s) with ExLlamaV3 vision model...")
        image_embeddings = []
        for img in images:
            embedding = vision_model.get_image_embeddings(tokenizer=tokenizer, image=img)
            image_embeddings.append(embedding)

        logger.info(f"Generated {len(image_embeddings)} ExLlamaV3 image embedding(s).")
        return image_embeddings
    except Exception as e:
        logger.error(f"Failed to generate image embeddings: {e}")
        return []


def inject_image_placeholders(text: str, image_embeddings: List[Any]) -> str:
    """Injects image placeholder text into the prompt string for ExLlamaV3."""
    if not image_embeddings:
        return text

    placeholders = "\n".join([ie.text_alias for ie in image_embeddings])
    return f"{placeholders}\n{text}"


def generate_with_images(prompt: str, images: List[Image.Image], **generate_params):
    """Generates text with images using the native ExLlamaV3 streaming method."""
    if not images or not is_multimodal_model():
        return None

    try:
        state = generate_params.copy()
        state['raw_images'] = images

        logger.info(f"Processing {len(images)} image(s) with native ExLlamaV3 generation.")
        result_generator = shared.model.generate_with_streaming(prompt, state)

        for response_text in result_generator:
            yield response_text
    except Exception as e:
        logger.error(f"Failed to generate with images (V3): {e}")
        return None


def generate_with_embeddings(prompt: str, image_embeddings: List[Any], **generate_params):
    """Generates text with pre-computed embeddings using the native ExLlamaV3 method."""
    if not image_embeddings or not is_multimodal_model():
        return None

    try:
        state = generate_params.copy()
        state['image_embeddings'] = image_embeddings

        logger.info(f"Processing {len(image_embeddings)} embedding(s) with native ExLlamaV3 generation.")
        result_generator = shared.model.generate_with_streaming(prompt, state)

        for response_text in result_generator:
            yield response_text
    except Exception as e:
        logger.error(f"Failed to generate with embeddings (V3): {e}")
        return None
