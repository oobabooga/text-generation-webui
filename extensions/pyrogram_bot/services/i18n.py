import yaml
from os.path import dirname

from typing import Callable

root_folder = dirname(dirname(__file__))

def get_i18n(lang = 'en') -> Callable[[str, dict[str, str]], str]:
  file_path = f"{root_folder}/i18n/{lang}.yml"

  dict_parsed = {}
  with open(file_path) as file:
    file_content = file.read()
    dict_parsed = yaml.load(file_content, Loader=yaml.FullLoader)

  def t(_, address: str, replaceble: dict[str, str] = {}) -> str:
    result = dict_parsed

    for i in address.split('.'):
      elem = result.get(i, None)

      if not elem:
        raise KeyError(address)

      result = elem
      if type(elem) == str:
        for before, after in replaceble.items():
          elem = elem.replace(before, str(after))
        return elem

  return t
