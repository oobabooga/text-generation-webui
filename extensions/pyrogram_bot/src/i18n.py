import os, json

def get_i18n(lang = 'en'):
  with open(f"{os.path.dirname(__file__)}/i18n.json") as file:
    return json.loads(file.read())[lang]