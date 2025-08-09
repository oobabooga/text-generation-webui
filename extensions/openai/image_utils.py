"""
Shared image processing utilities for multimodal support.
Used by both ExLlamaV3 and llama.cpp implementations.
"""
import base64
import io
from typing import Any, List, Tuple

from PIL import Image

from modules.logging_colors import logger


def convert_pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    # Save image to an in-memory bytes buffer in PNG format
    image.save(buffered, format="PNG")
    # Encode the bytes to a base64 string
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
    Returns: A tuple of (text_content, list_of_pil_images).
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
                        logger.info("Successfully loaded external image from URL")
                    except Exception as e:
                        logger.warning(f"Failed to fetch external image: {e}")
                else:
                    logger.warning(f"Unsupported image URL format: {image_url[:70]}...")

        return ' '.join(text_parts), images

    return str(content), []


def convert_image_attachments_to_pil(image_attachments: List[dict]) -> List[Image.Image]:
    """Convert webui image_attachments format to PIL Images."""
    pil_images = []
    for attachment in image_attachments:
        if attachment.get('type') == 'image' and 'image_data' in attachment:
            try:
                image = decode_base64_image(attachment['image_data'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                pil_images.append(image)
            except Exception as e:
                logger.warning(f"Failed to process image attachment: {e}")
    return pil_images


def convert_openai_messages_to_images(messages: List[dict]) -> List[Image.Image]:
    """Convert OpenAI messages format to PIL Images."""
    all_images = []
    for message in messages:
        if isinstance(message, dict) and 'content' in message:
            _, images = process_message_content(message['content'])
            all_images.extend(images)
    return all_images
