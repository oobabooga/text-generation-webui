import gradio as gr
from deep_translator import GoogleTranslator, DeeplTranslator, LibreTranslator

import os
import json

path_settings_json =  "extensions/google_translate/settings.json"

params = {
    "language string": "ja",
    "translator": "GoogleTranslator",
    "custom_url": "",

}

language_codes = {'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Nyanja (Chichewa)': 'ny', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese (Portugal, Brazil)': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala (Sinhalese)': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tagalog (Filipino)': 'tl', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
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

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    if params['translator'] == "GoogleTranslator":
        return GoogleTranslator(target=params['language string'], source='en').translate(string)
    if params['translator'] == "DeeplTranslator":
        return DeeplTranslator(target=params['language string'], source='en').translate(string)
    if params['translator'] == "LibreTranslator":
        #print("LibreTranslator using output_modifier")
        custom_url = params['custom_url']
        if custom_url == "": custom_url = "https://translate.argosopentech.com/"
        return LibreTranslator(target=params['language string'], source='en', custom_url = custom_url).translate(string)


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string


def ui():
    # Finding the language name from the language code to use as the default value
    language_name = list(language_codes.keys())[list(language_codes.values()).index(params['language string'])]

    # Gradio elements
    language = gr.Dropdown(value=language_name, choices=[k for k in language_codes], label='Language')

    # Event functions to update the parameters in the backend
    language.change(lambda x: params_update({"language string": language_codes[x]}), language, None)

    # DeeplTranslator not work for now; api key required
    translator = gr.Dropdown(value=params['translator'], choices=["GoogleTranslator", "LibreTranslator"], label='Translator')

    translator.change(lambda x: params_update({"translator": x}), translator, None)

    custom_url = gr.Textbox(value=params['custom_url'], label='Custom URL for translation API (affect LibreTranslator now)')

    custom_url.change(lambda x: params_update({"custom_url": x}), custom_url, None)



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