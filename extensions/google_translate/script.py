import gradio as gr
from deep_translator import GoogleTranslator, DeeplTranslator, LibreTranslator

import os
import json

from modules.text_generation import encode, generate_reply

path_settings_json =  "extensions/google_translate/settings.json"

params = {
    "language string": "en",
    "translator": "GoogleTranslator",
    "custom_url": "",
    "is_translate_user": True,
    "is_translate_system": True,
    "is_add_system_orig": False,

}

language_codes = {'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Nyanja (Chichewa)': 'ny', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese (Portugal, Brazil)': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala (Sinhalese)': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tagalog (Filipino)': 'tl', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}

# tpl for local
tpl = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
tpl += "### Instruction:\nTranslate phrase from {0} to {1}\n"
tpl += "### Input:\n{2}\n"
tpl += "### Response:"
tpl_alpaca = tpl


def ui():
    # Finding the language name from the language code to use as the default value
    language_name = list(language_codes.keys())[list(language_codes.values()).index(params['language string'])]

    # Gradio elements
    language = gr.Dropdown(value=language_name, choices=[k for k in language_codes], label='Language')

    # Event functions to update the parameters in the backend
    language.change(lambda x: params_update({"language string": language_codes[x]}), language, None)

    # DeeplTranslator not work for now; api key required
    translator = gr.Dropdown(value=params['translator'], choices=["GoogleTranslator", "LibreTranslator", "LocalAlpaca"],
                             label='Translator')

    translator.change(lambda x: params_update({"translator": x}), translator, None)

    custom_url = gr.Textbox(value=params['custom_url'],
                            label='Custom URL for translation API (affect LibreTranslator now)')

    custom_url.change(lambda x: params_update({"custom_url": x}), custom_url, None)

    is_translate_user = gr.Checkbox(value=params['is_translate_user'], label='Translate user input')

    is_translate_user.change(lambda x: params_update({"is_translate_user": x}), is_translate_user, None)

    is_translate_system = gr.Checkbox(value=params['is_translate_system'], label='Translate system output')

    is_translate_system.change(lambda x: params_update({"is_translate_system": x}), is_translate_system, None)

    is_add_system_orig = gr.Checkbox(value=params['is_add_system_orig'], label='Add system origin output to translation')

    is_add_system_orig.change(lambda x: params_update({"is_add_system_orig": x}), is_translate_system, None)


def language_code_to_lang(langcode:str):
    for i in language_codes.keys():
        if language_codes[i] == langcode:
            return i
    return ""

def local_translator(from_lang:str,to_lang:str,string:str,prompt_tpl:str,body:dict = None):
    prompt = prompt_tpl.format(
        language_code_to_lang(from_lang),
        language_code_to_lang(to_lang),
        string
    )

    print("LocalTranslation prompt:",prompt)

    if body is None:
        body = {}

    generate_params = {
        'max_new_tokens': int(body.get('max_length', 200)),
        'do_sample': bool(body.get('do_sample', True)),
        'temperature': float(body.get('temperature', 0.5)),
        'top_p': float(body.get('top_p', 0.2)),
        'typical_p': float(body.get('typical', 1)),
        'repetition_penalty': float(body.get('rep_pen', 1.1)),
        'encoder_repetition_penalty': 1,
        'top_k': int(body.get('top_k', 30)),
        'min_length': int(body.get('min_length', 0)),
        'no_repeat_ngram_size': int(body.get('no_repeat_ngram_size', 0)),
        'num_beams': int(body.get('num_beams', 1)),
        'penalty_alpha': float(body.get('penalty_alpha', 0)),
        'length_penalty': float(body.get('length_penalty', 1)),
        'early_stopping': bool(body.get('early_stopping', True)),
        'seed': int(body.get('seed', -1)),
        'add_bos_token': int(body.get('add_bos_token', True)),
        'custom_stopping_strings': body.get('custom_stopping_strings', []),
        'truncation_length': int(body.get('truncation_length', 2048)),
        'ban_eos_token': bool(body.get('ban_eos_token', False)),
        'skip_special_tokens': bool(body.get('skip_special_tokens', True)),
    }

    generator = generate_reply(
        prompt,
        generate_params,
    )

    answer = ''
    for a in generator:
        if isinstance(a, str):
            answer = a
        else:
            answer = a[0]

    print("LocalTranslation answer:", answer)

    return answer


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['is_translate_user']: return string # no translation needed
    if params['language string'] == "en": return string # no translation needed

    if params['translator'] == "GoogleTranslator":
        #print("GoogleTranslator using")
        return GoogleTranslator(source=params['language string'], target='en').translate(string)
    if params['translator'] == "DeeplTranslator":
        #print("Deepl using")
        return DeeplTranslator(source=params['language string'], target='en').translate(string)
    if params['translator'] == "LibreTranslator":
        #print("LibreTranslator using input_modifier")
        custom_url = params['custom_url']
        if custom_url == "": custom_url = "https://translate.argosopentech.com/"
        return LibreTranslator(source=params['language string'], target='en', custom_url = params['custom_url']).translate(string)
    if params['translator'] == "LocalAlpaca":
        #print("GoogleTranslator using")
        #return GoogleTranslator(source=params['language string'], target='en').translate(string)
        return local_translator(params['language string'],'en',string,tpl_alpaca)

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    if not params['is_translate_system']: return string  # no translation needed
    if params['language string'] == "en": return string  # no translation needed

    res = ""
    if params['translator'] == "GoogleTranslator":
        res = GoogleTranslator(target=params['language string'], source='en').translate(string)
    if params['translator'] == "DeeplTranslator":
        res = DeeplTranslator(target=params['language string'], source='en').translate(string)
    if params['translator'] == "LibreTranslator":
        #print("LibreTranslator using output_modifier")
        custom_url = params['custom_url']
        if custom_url == "": custom_url = "https://translate.argosopentech.com/"
        res = LibreTranslator(target=params['language string'], source='en', custom_url = custom_url).translate(string)
    if params['translator'] == "LocalAlpaca":
        #print("GoogleTranslator using")
        #return GoogleTranslator(source=params['language string'], target='en').translate(string)
        res = local_translator('en',params['language string'],string,tpl_alpaca)

    if params['is_add_system_orig']: # add original response
        res += "\n\n_({0})_".format(string)

    return res

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string






def params_update(upd):
    global params
    params.update(upd)
    save_settings()

def save_settings():
    global params


    with open(path_settings_json, 'w') as f:
        json.dump(params, f, indent=2)


def load_settings():
    global params

    try:
        with open(path_settings_json, 'r') as f:
            # Load the JSON data from the file into a Python dictionary
            data = json.load(f)

        if data:
            params = {**params, **data} # mix them, this allow to add new params seamlessly

    except FileNotFoundError:
        #memory_settings = {"position": "Before Context"}
        pass

    #return memory_settings["position"]
def setup():
    load_settings()