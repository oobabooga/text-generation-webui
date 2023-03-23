from os import getcwd, listdir
from os.path import dirname, join

from modules import shared

from .i18n import get_i18n

root_path = dirname(dirname(__file__))
t = get_i18n()

def get_chars() -> list[str]:
  char_path = join(getcwd(), 'characters')

  chars = []

  for char_filename in listdir(char_path):
    if not '.json' in char_filename:
      continue

    char = char_filename.replace('.json', '')
    if char == shared.character:
      char = f"**{char}**"

    chars.append(char)

  chars.sort()

  return chars

def set_char(input_char_name: str) -> bool:
  char_filename = input_char_name+'.json'
  char_path = join(getcwd(), 'characters')
  char_exists = char_filename in listdir(char_path)

  status = 'failed'
  if char_exists:
    shared.character = input_char_name
    status = 'successfull'

  return status
