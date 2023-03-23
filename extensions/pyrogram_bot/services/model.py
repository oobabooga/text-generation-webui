from os import getcwd, listdir
from os.path import dirname, join, isdir

from modules import shared
from server import load_model_wrapper

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

def set_model(input_model_name: str) -> bool:
  model_path = join(getcwd(), 'models')
  model_exists = input_model_name in listdir(model_path)

  status = False
  if model_exists:
    load_model_wrapper(input_model_name)
    status = True

  return status
