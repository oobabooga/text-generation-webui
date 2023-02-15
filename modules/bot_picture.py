import requests
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# raw_image = Image.open('/tmp/istockphoto-470604022-612x612.jpg').convert('RGB')
def caption_image(raw_image):
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(out[0], skip_special_tokens=True)
