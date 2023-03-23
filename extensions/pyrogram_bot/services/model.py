from os import getcwd, listdir
from os.path import dirname, join, isdir

from modules import shared

from .i18n import get_i18n

root_path = dirname(dirname(__file__))
t = get_i18n()

def get_models() -> list[str]:
  model_path = join(getcwd(), 'models')

  models = []

  for model in listdir(model_path):
    if not isdir(join(model_path, model)):
      continue

    if model == shared.model_name:
      model = f"**{model}**"

    models.append(model)

  models.sort()

  return models

def set_model(input_char_name: str) -> bool:
  char_filename = input_char_name+'.json'
  char_path = join(getcwd(), 'characters')
  char_exists = char_filename in listdir(char_path)

  status = False
  if char_exists:
    shared.character = input_char_name
    status = True

  return status
