from yaml import load as parse
from os.path import dirname

root_folder = dirname(dirname(__file__))

def get_i18n(lang = 'en'):
  with open(f"{root_folder}/i18n/{lang}.yml") as file:
    return parse(file.read())
