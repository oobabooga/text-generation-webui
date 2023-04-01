from os import getcwd, listdir
from os.path import join

from pyrogram.types import Message

from modules import shared

def prepare_message() -> list[str]:
  char_path = join(getcwd(), 'characters')

  chars = []

  for char_filename in listdir(char_path):
    if not '.json' in char_filename:
      continue

    char = char_filename.replace('.json', '')
    if char == shared.character:
      char = f"➢ **{char}**"

    chars.append(char)

  chars.sort()

  return chars

async def action(app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  answer = app.t('character.index') + '\n'.join(prepare_message())

  await new_msg.edit(answer)
  msg.stop_propagation()
