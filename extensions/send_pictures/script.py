import base64
from io import BytesIO
import os

import gradio as gr
import tensorflow as tf
from modules import chat, shared
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTFeatureExtractor, AutoTokenizer, \
    VisionEncoderDecoderModel
import time

# It does take some effort to install DeepDanbooru, so not everyone will want or need it.
try:
    import deepdanbooru as dd
except ImportError:
    deepdanbooru_installed = False
    print('Could not import deepdanbooru module - proceeding without it')
else:
    deepdanbooru_installed = True

# Parameters which can be customized in settings.json of webui
params = {
    'networks': [],
    'segment': False,
    'booru_model_path': 'repositories/DeepDanbooru/deepdanbooru/model/v3/',
    'booru_h5': 'model-resnet_custom_v3.h5',
    'booru_tags': 'tags.txt',
    'booru_cutoff': 0.75
}

# Indicate img2txt model from Huggingface using these strings ("user/model" syntax)
model_str = "Salesforce/blip-image-captioning-large"
model_str_alt = "nlpconnect/vit-gpt2-image-captioning"
booru_str = "DeepDanbooru"

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


# Function to initialize DeepBooru model
def booru_init():
    with tf.device('/cpu:0'):
        booru_tags = dd.data.load_tags(params['booru_model_path'] + params['booru_tags'])
        booru_model = tf.keras.models.load_model(params['booru_model_path'] + params['booru_h5'], compile=False)

    return booru_model, booru_tags


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
        output_ids = model_alt.generate(pixel_values, max_length=24, num_beams=4,
                                        return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


# Function for BLIP captioning
def caption_image(raw_image):
    inputs = processor(raw_image.convert('RGB'), return_tensors="pt").to("cpu", torch.float32)
    inputs_alt = predict(raw_image.convert('RGB'))
    out = model.generate(**inputs, max_new_tokens=24)

    return processor.decode(out[0], skip_special_tokens=True)


# Function for tagging with DeepDanbooru
def tags(raw_image):
    raw_image.save("send_pictures_cache.png")

    no_tags_string = 'is something that could not be described'
    h5_path = params['booru_model_path'] + params['booru_h5']
    tags_path = params['booru_model_path'] + params['booru_tags']
    path_fail = False

    # Check that required DeepBooru model files are in place
    if not os.path.isdir(params['booru_model_path']):
        print('No such directory ' + params['booru_model_path'])
        path_fail = True
    else:
        if not os.path.isfile(h5_path):
            print('*.h5 file not found: ' + params['booru_h5'] +
                  '. Make sure it is in the same directory as the model.')
            path_fail = True
        if not os.path.isfile(tags_path):
            print('Tags file not found: ' + params['booru_tags'] +
                  '. Make sure it is in the same directory as the model.')
            path_fail = True

    if path_fail:
        print('DeepBooru tagging could not be initiated')
        return no_tags_string

    with tf.device('/cpu:0'):
        taglist = ''
        booru_model, booru_tags = booru_init()
        dd_out = dd.commands.evaluate_image("send_pictures_cache.png", booru_model, booru_tags, params['booru_cutoff'])

        for tag, score in dd_out:
            if taglist != '':
                taglist += ', '
            taglist += f'{tag}'

    return taglist


# Return descriptions based on the selected inference method
def infer(image):
    networks = params['networks']
    out = ''

    if model_str in networks:
        out += caption_image(image)

    if model_str_alt in networks:
        if out != '':
            out += ', or maybe '
        out += predict(image)[0]

    if booru_str in networks:
        if out != '':
            out += '; '
        out += f'described though tags as follows: {tags(image)}'

    return out


# Function determining UI output and the text passed to the LLM
def generate_chat_picture(picture, name1, name2):
    networks = params['networks']
    do_segment = params['segment']

    # What the user sees as the message they sent in chat
    visible_text = f'*{name1} sent {name2} a picture*'
    # visible_text = f'<img src="data:image/jpeg;base64,{img_str}" alt="{text}">\n'
    # Commented out because HTML is appearing as plain text in the current build

    # Prepare the plain language prompt wrapper for the captions with an intro
    text = f'*{name1} sent {name2} a picture. '

    # Escape if no inference options are selected
    if not opts_selected(networks):
        return f' Unfortunately, {name2} was not able to receive it.*', f'*Image not sent. Please select at least one inference network to use.*'

    # Captioning execution timer start
    start = time.time()

    if do_segment:
        text += f'{name2} sees indistinct impressions of different parts of that picture. They may conflict, but things that overlap in these impressions are probably actually in the picture {name2} saw:\n'

        # Segment image into two horizontal strips and two vertical strips, storing the segments in a unidimensional array of images
        splitImage = image_crop(picture, 1, 2)
        splitImage.extend(image_crop(picture, 2, 1))

        # Produce and merge in segment captions wrapped in plain language.
        #      TODO: Consider procedural generation to enable an arbitrary number of horizontal and vertical strips.
        #            This would require optimization of the plain language prompt wrapper for scalability.
        text += f'The top of the picture makes {name2} think the whole picture is {infer(splitImage[0])}.\n'
        text += f'The bottom of the picture makes {name2} think the whole picture is {infer(splitImage[1])}.\n'
        text += f'The left of the picture makes {name2} think the whole picture is {infer(splitImage[2])}.\n'
        text += f'The right of the picture makes {name2} think the whole picture is {infer(splitImage[3])}.\n'

        # Alternative code for splitting the image into an arbitrary number of non-overlapping rectangles - gives inferior performance in informal testing
        # for i in range(len(splitImage)):
        #   text += f'{i+1}. {caption_image(splitImage[i])}; {predict(splitImage[i])[0]}. \n'

    # Whole image caption
    text += f'\nFrom a distance, {name2} thinks the whole picture looks like {infer(picture)}. {name2} '

    # Wrapper branching depending on which networks and whether segmentation were used
    if model_str in networks or model_str_alt in networks or do_segment:
        text += 'puts all this information together in order to understand the whole picture they were sent'
        if model_str in networks:
            text += ', and '
        else:
            text += '. '
    if model_str in networks:
        text += 'strings the tags together into a full english description of the image. '

    # Wrapper outro
    text += f'{name2} then begins to describe the picture to {name1} without making up any details.*'

    # Captioning execution timer stop and print
    caption_time = time.time() - start
    print(f'Image captioning prompt generated in {round(caption_time, 2)} seconds')

    # Lower the resolution of sent images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
    picture.thumbnail((300, 300))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return text, visible_text


# Function checking for an inference option configuration
def opts_selected(param_list):
    if len(param_list) > 0 and param_list != ['']:
        return True
    else:
        return False

    # Function defining UI behavior


def ui():
    if deepdanbooru_installed:
        networks = gr.CheckboxGroup(
            [model_str, model_str_alt, booru_str],
            value=params['networks'],
            label='Inference Networks',
            info='Infer image contents from a combination of tagging and captioning methods.'
        )
    else:
        networks = gr.CheckboxGroup(
            [model_str, model_str_alt],
            value=params['networks'],
            label='Inference Networks',
            info='Infer image contents from a combination of tagging and captioning methods.'
        )

    do_segment = gr.Checkbox(
        value=False,
        label='Use image strip analysis (~5x slower and uses 3-5x more context. Usually gives better output)'
    )

    picture_select = gr.Image(
        label='Send a picture',
        type='pil',
        visible=True
    )

    with gr.Accordion(label="DeepDanbooru Settings", visible=deepdanbooru_installed):
        booru_args = [
            gr.Slider(
                label="Score cutoff value",
                maximum=1,
                value=params['booru_cutoff'],
                step=0.05,
                info='Tags scoring below this value will not be included in the prompt.'
            ),
            gr.Textbox(
                label="Path to DeepDanbooru Model",
                info="Relative to text-generation-webui",
                lines=1,
                value=params['booru_model_path'],
                interactive=True
            ),
            gr.Textbox(
                label="Model",
                info="Name of model file (*.h5)",
                lines=1,
                value=params['booru_h5'],
                interactive=True
            ),
            gr.Textbox(
                label="Tags",
                info="Name of tags file (*.txt)",
                lines=1,
                value=params['booru_tags'],
                interactive=True
            )
        ]

    # function_call = 'chat.cai_chatbot_wrapper' if shared.args.chat else 'chat.chatbot_wrapper'
    function_call = 'chat.cai_chatbot_wrapper' if shared.args.chat else 'chat.chatbot_wrapper'

    # Event functions to update the parameters in the backend
    networks.change(lambda x: params.update({"networks": x}), networks, None)
    do_segment.change(lambda x: params.update({"segment": x}), do_segment, None)
    booru_args[0].change(lambda x: params.update({"booru_cutoff": x}), booru_args[0], None)
    booru_args[1].change(lambda x: params.update({"booru_model_path": x}), booru_args[1], None)
    booru_args[2].change(lambda x: params.update({"booru_h5": x}), booru_args[2], None)
    booru_args[3].change(lambda x: params.update({"booru_tags": x}), booru_args[3], None)

    # Prepare the hijack with custom inputs
    picture_select.upload(lambda picture, name1, name2: input_hijack.update(
        {"state": True, "value": generate_chat_picture(picture, name1, name2)}),
                          [picture_select, shared.gradio['name1'], shared.gradio['name2']], None)

    # Call the generation function
    picture_select.upload(eval(function_call), shared.input_params, shared.gradio['display'],
                          show_progress=shared.args.no_stream)

    # Clear the picture from the upload field
    picture_select.upload(lambda: None, [], [picture_select], show_progress=False)
