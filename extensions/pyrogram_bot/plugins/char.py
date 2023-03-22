from os import getcwd, listdir
from os.path import dirname, join

from modules import shared

from ..src.i18n import get_i18n

root_path = dirname(dirname(__file__))
t = get_i18n()

try:
  from pyrogram import Client, filters
  from pyrogram.types import Message
except:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', {'/path/': root_path}))

@Client.on_message(filters.command(["characters"]))
async def index(_: Client, msg: Message) -> None:
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

  answer = t('character.index') + '\n'.join(chars)
  await msg.reply(answer)
  msg.stop_propagation()

@Client.on_message(filters.command(["character"]))
async def get(_: Client, msg: Message) -> None:
  await msg.reply(t('character.get', {'/name/': shared.character}))
  msg.stop_propagation()

@Client.on_message(filters.command(["set_character"]))
async def put(_: Client, msg: Message) -> None:
  char_filename = msg.command[1]+'.json'
  char_path = join(getcwd(), 'characters')
  char_exists = char_filename in listdir(char_path)

  status = 'failed'
  if char_exists:
    shared.character = msg.command[1]
    status = 'succesful'

  answer = t(f'character.put.{status}', {'/name/': shared.character})
  await msg.reply(answer)
  msg.stop_propagation()
