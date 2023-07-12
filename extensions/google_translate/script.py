import gradio as gr
from deep_translator import GoogleTranslator
import os
import re
import uuid

params = {
    "activate": True,
    "language string": "ko",
}

language_codes = {'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Traditional)': 'zh-TW', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Nyanja (Chichewa)': 'ny', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese (Portugal, Brazil)': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala (Sinhalese)': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tagalog (Filipino)': 'tl', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}

def modify_string(string, source, target):
    pattern = re.compile(r'```(.*?)\n(.*?)```', re.DOTALL)
    blocks = [(str(uuid.uuid4()), m.group(1), m.group(2)) for m in re.finditer(pattern, string)]
    string_without_blocks = string
    for uuid_str, _, _ in blocks:
        string_without_blocks = re.sub(pattern, uuid_str, string_without_blocks, count=1)
    translated_string = GoogleTranslator(source=source, target=target).translate(string_without_blocks)
    for uuid_str, lang, block in blocks:
        # Remove leading and trailing whitespaces from each block
        block = block.strip()
        translated_string = translated_string.replace(uuid_str, '```' + lang + '\n' + block + '\n```')
    return translated_string


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['activate']:
        return string

    return modify_string(string, params['language string'], 'en')


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    if not params['activate']:
        return string

    return modify_string(string, 'en', params['language string'])


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string


def ui():
    params['language string'] = read_language_code()


    # Finding the language name from the language code to use as the default value
    language_name = list(language_codes.keys())[list(language_codes.values()).index(params['language string'])]

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate translation')

    with gr.Row():
        language = gr.Dropdown(value=language_name, choices=[k for k in language_codes], label='Language')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    # language.change(lambda x: params.update({"language string": language_codes[x]}), language, None)
    language.change(lambda x: (new_language_code := language_codes[x], params.update({"language string": new_language_code}), write_language_code(new_language_code)), language, None)


def read_language_code(filename="setting/latest_use_language.txt"):
    try:
        with open(filename, "r") as file:
            language_code = file.read().strip()
        return language_code
    except FileNotFoundError:
        print(f"Cannot find the file {filename}.")
        return 'ko'

def write_language_code(language_code, filename="setting/latest_use_language.txt"):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as file:
        file.write(language_code)