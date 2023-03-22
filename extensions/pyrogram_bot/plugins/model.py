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

@Client.on_message(filters.command(["models"]))
async def index(_: Client, msg: Message) -> None:
  models_dir = os.getcwd() + '/models'
  models = []

  for model in os.listdir(models_dir):
    if os.path.isdir(os.path.join(models_dir, model)):
      if model == shared.model_name:
        models.append(f"**{model}**")
      else:
        models.append(model)
        
  answer = t['model']['index'] + '\n'.join(models)
  await msg.reply(answer)

  raise StopPropagation


@Client.on_message(filters.command(["model"]))
async def get(_: Client, msg: Message) -> None:
  answer = t['model']['get'].replace('/name/', shared.model_name)
  await msg.reply(answer)

  raise StopPropagation

@Client.on_message(filters.command(["set_model"]))
async def put(_: Client, msg: Message) -> None:

  if msg.command[1]+'.json' in os.listdir(os.getcwd() + '/characters'):
    shared.character = msg.command[1]
    status = 'succesful'
  else:
    status = 'failed'
  
  answer = t['model']['put'][status].replace('/name/', shared.character)
  await msg.reply(answer)

  raise StopPropagation