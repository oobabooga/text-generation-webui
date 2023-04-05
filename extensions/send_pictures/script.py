import base64
from io import BytesIO

import gradio as gr
import modules.chat as chat
import modules.shared as shared
import torch
import requests
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

# If 'state' is True, will hijack the next chat generation with
# custom input text given by 'value' in the format [text, visible_text]
input_hijack = {
    'state': False,
    'value': ["", ""]
}

# Indicate img2txt model from Huggingface using this string ("user/model" syntax)

model_str = "Salesforce/blip-image-captioning-large"
model_str_alt = "nlpconnect/vit-gpt2-image-captioning"

processor = BlipProcessor.from_pretrained(model_str)
model = BlipForConditionalGeneration.from_pretrained(model_str, torch_dtype=torch.float32).to("cpu")

feature_extractor = ViTFeatureExtractor.from_pretrained(model_str_alt)
tokenizer = AutoTokenizer.from_pretrained(model_str_alt)
model_alt = VisionEncoderDecoderModel.from_pretrained(model_str_alt, torch_dtype=torch.float32).to("cpu")

model_alt.eval()

# Function for vit-gpt2 prediction

def predict(image):

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model_alt.generate(pixel_values, max_length=24, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

def caption_image(raw_image):
    inputs = processor(raw_image.convert('RGB'), return_tensors="pt").to("cpu", torch.float32)
    inputs_alt = predict(raw_image.convert('RGB'))
    out = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_chat_picture(picture, name1, name2):
    # merge the two predicted captions for the final text
    text = f'*{name1} sends {name2} a picture that contains {caption_image(picture)}, alternatively appearing to be {predict(picture)[0]}*'
    # lower the resolution of sent images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
    picture.thumbnail((300, 300))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible_text = f'<img src="data:image/jpeg;base64,{img_str}" alt="{text}">'
    return text, visible_text

def ui():
    picture_select = gr.Image(label='Send a picture', type='pil')

    # Prepare the hijack with custom inputs
    picture_select.upload(lambda picture, name1, name2: input_hijack.update({"state": True, "value": generate_chat_picture(picture, name1, name2)}), [picture_select, shared.gradio['name1'], shared.gradio['name2']], None)

    # Call the generation function
    picture_select.upload(chat.cai_chatbot_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)

    # Clear the picture from the upload field
    picture_select.upload(lambda : None, [], [picture_select], show_progress=False)
