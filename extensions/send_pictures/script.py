import base64
from io import BytesIO

import gradio as gr
import modules.chat as chat
import modules.shared as shared
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# If 'state' is True, will hijack the next chat generation with
# custom input text given by 'value' in the format [text, visible_text]
input_hijack = {
    'state': False,
    'value': ["", ""]
}

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float32).to("cpu")

def caption_image(raw_image):
    inputs = processor(raw_image.convert('RGB'), return_tensors="pt").to("cpu", torch.float32)
    out = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_chat_picture(picture, name1, name2):
    text = f'*{name1} sends {name2} a picture that contains the following: "{caption_image(picture)}"*'
    # lower the resolution of sent images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
    picture.thumbnail((300, 300))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible_text = f'<img src="data:image/jpeg;base64,{img_str}" alt="{text}">'
    return text, visible_text

def ui():
    picture_select = gr.Image(label='Send a picture', type='pil')

    function_call = 'chat.cai_chatbot_wrapper' if shared.args.cai_chat else 'chat.chatbot_wrapper'

    # Prepare the hijack with custom inputs
    picture_select.upload(lambda picture, name1, name2: input_hijack.update({"state": True, "value": generate_chat_picture(picture, name1, name2)}), [picture_select, shared.gradio['name1'], shared.gradio['name2']], None)

    # Call the generation function
    picture_select.upload(eval(function_call), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)

    # Clear the picture from the upload field
    picture_select.upload(lambda : None, [], [picture_select], show_progress=False)
