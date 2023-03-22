import os

from modules import shared

from ..src.i18n import get_i18n

path = os.path.dirname(__file__)

t = get_i18n()

try:
  from pyrogram import Client, StopPropagation, filters
  from pyrogram.types import Message
except:
  raise Exception(t['error']['dependencies_not_installed'].replace('/path/', path))

@Client.on_message(filters.command(["characters"]))
async def index(_: Client, msg: Message) -> None:
  chars = []
  for file in os.listdir(os.getcwd() + '/characters'):
    if file.endswith('.json'):
      char = (
        file
          .replace('.json', '')
          .replace('-', ' ')
          .replace('Z ', 'NSFW ')
      )
      chars.append(char)
      
  chars.sort()

  answer = t['character']['index'] + '\n'.join(chars)
  await msg.reply(answer)
  
  raise StopPropagation

@Client.on_message(filters.command(["character"]))
async def get(_: Client, msg: Message) -> None:
  answer = t['character']['get'].replace('/name/', shared.character)
  await msg.reply(answer)

  raise StopPropagation

@Client.on_message(filters.command(["set_character"]))
async def put(_: Client, msg: Message) -> None:

  if msg.command[1]+'.json' in os.listdir(os.getcwd() + '/characters'):
    shared.character = msg.command[1]
    status = 'succesful'
  else:
    status = 'failed'
  
  answer = t['character']['put'][status].replace('/name/', shared.character)
  await msg.reply(answer)

  raise StopPropagation
