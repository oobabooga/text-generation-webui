import base64
from io import BytesIO

import numpy as np
import gradio as gr
import modules.chat as chat
import modules.shared as shared
import torch
import requests
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import time


# Indicate img2txt model from Huggingface using these strings ("user/model" syntax)
model_str = "Salesforce/blip-image-captioning-large"
model_str_alt = "nlpconnect/vit-gpt2-image-captioning"

# If 'state' is True, will hijack the next chat generation with
# custom input text given by 'value' in the format [text, visible_text]
input_hijack = {
    'state': False,
    'value': ["", ""]
}

# Initialize BLIP model
processor = BlipProcessor.from_pretrained(model_str)
model = BlipForConditionalGeneration.from_pretrained(model_str, torch_dtype=torch.float32).to("cpu")

# Initialize Vit-GPT2 model
feature_extractor = ViTFeatureExtractor.from_pretrained(model_str_alt)
tokenizer = AutoTokenizer.from_pretrained(model_str_alt)
model_alt = VisionEncoderDecoderModel.from_pretrained(model_str_alt, torch_dtype=torch.float32).to("cpu")
model_alt.eval() 


# Function for image cropping
def image_crop(image, x_pieces, y_pieces):
    
    sections = []
    image_width, image_height = image.size
    height = image_height // y_pieces
    width = image_width // x_pieces
    
    for i in range(0, y_pieces):
        for j in range(0, x_pieces):
            box = (j * width, i * height, (j + 1)
                   * width, (i + 1) * height)
            section = image.crop(box)
            sections.append(section)
            
    return sections


# Function for vit-gpt2 captioning
def predict(image):

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        output_ids = model_alt.generate(pixel_values, max_length=24, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


# Function for BLIP captioning
def caption_image(raw_image):

    inputs = processor(raw_image.convert('RGB'), return_tensors="pt").to("cpu", torch.float32)
    inputs_alt = predict(raw_image.convert('RGB'))
    out = model.generate(**inputs, max_new_tokens=24)
    
    return processor.decode(out[0], skip_special_tokens=True)


# Function determining UI output and the text passed to the LLM
def generate_chat_picture(picture, name1, name2):
    
    # What the user sees as the message they sent in chat
    visible_text =f'*{name1} sent {name2} a picture*'
    #visible_text = f'<img src="data:image/jpeg;base64,{img_str}" alt="{text}">\n'  # Commented out because HTML is appearing as plain text in the current build
    
    # Captioning execution timer start 
    start = time.time()
    
    # Prepare the plain language prompt wrapper for the captions
    text = f'*{name1} sent {name2} a picture. {name2} sees indistinct impressions of different parts of that picture. They may conflict, but things that overlap in these impressions are probably actually in the picture {name2} saw:\n'
    
    # Segment image into two horizontal strips and two vertical strips, storing the segments in a unidimensional array of images
    splitImage = image_crop(picture,1,2)
    splitImage.extend(image_crop(picture,2,1))
    
    # Produce and merge in segment captions wrapped in plain language.     
    #      TODO: Consider procedural generation to enable an arbitrary number of horizontal and vertical strips.
    #            This would require optimization of the plain language prompt wrapper for scalability.
    text += f'The top of the picture makes {name2} think the whole picture is {caption_image(splitImage[0])}; {predict(splitImage[0])[0]}.\n'
    text += f'The bottom of the picture makes {name2} think the whole picture is {caption_image(splitImage[1])}; {predict(splitImage[1])[0]}.\n'
    text += f'The left of the picture makes {name2} think the whole picture is {caption_image(splitImage[2])}; {predict(splitImage[2])[0]}.\n'
    text += f'The right of the picture makes {name2} think the whole picture is {caption_image(splitImage[3])}; {predict(splitImage[3])[0]}.\n'
    
    # Alternative code for splitting the image into an arbitrary number of non-overlapping rectangles - gives inferior performance in informal testing
    # for i in range(len(splitImage)):
    #   text += f'{i+1}. {caption_image(splitImage[i])}; {predict(splitImage[i])[0]}. \n'
     
    # Whole image captions from two sources 
    text += f'\nFrom a distance, {name2} thinks the whole picture looks like {caption_image(picture)}, or maybe {predict(picture)[0]}. \n{name2} puts all this information together in order to understand the whole picture they were sent, then begin to describe the picture to {name1}.*'
    
    # Captioning execution timer stop and print
    caption_time = time.time()-start
    print(f'Image captioning prompt generated in {round(caption_time,2)} seconds')
    
    # Lower the resolution of sent images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
    picture.thumbnail((300, 300))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return text, visible_text


# Function defining UI behavior
#      TODO: UI element to select different captioning methods   
def ui():

    picture_select = gr.Image(label='Send a picture', type='pil')
    function_call = 'chat.cai_chatbot_wrapper' if shared.args.cai_chat else 'chat.chatbot_wrapper'

    # Prepare the hijack with custom inputs
    picture_select.upload(lambda picture, name1, name2: input_hijack.update({"state": True, "value": generate_chat_picture(picture, name1, name2)}), [picture_select, shared.gradio['name1'], shared.gradio['name2']], None)

    # Call the generation function
    picture_select.upload(eval(function_call), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)

    # Clear the picture from the upload field
    picture_select.upload(lambda : None, [], [picture_select], show_progress=False)
