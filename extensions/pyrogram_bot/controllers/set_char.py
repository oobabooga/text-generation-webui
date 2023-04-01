from os import getcwd, listdir
from os.path import join

from pyrogram.types import Message

from modules import shared, chat

def prepare_message(char_name: str) -> bool:
  char_path = join(getcwd(), 'characters')
  char_exists = char_name+'.json' in listdir(char_path)

  if not char_exists: return False

  char = chat.load_character(char_name, 'You', 'Char')
  shared.settings.update({
    'name2': char[0],
    'context': char[1],
  })

  return True

async def action(app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  status = 'succesful' if prepare_message(msg.command[1]) else 'failed'
  answer = app.t(f'model.put.{status}', {'/name/': shared.character})

  await new_msg.edit(answer)
  msg.stop_propagation()
