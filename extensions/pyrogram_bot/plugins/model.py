from os import getcwd, listdir
from os.path import dirname, isdir, join

from modules import shared

from ..src.i18n import get_i18n

root_path = dirname(dirname(__file__))
t = get_i18n()

try:
  from pyrogram import Client, filters
  from pyrogram.types import Message
except:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', {'/path/': root_path}))

@Client.on_message(filters.command(["models"]))
async def index(_: Client, msg: Message) -> None:
  model_path = join(getcwd(), 'models')
  print('model_path', model_path, flush=True)

  models = []

  for model in listdir(model_path):
    if not isdir(join(model_path, model)):
      continue

    if model == shared.model_name:
      model = f"**{model}**"

    models.append(model)

  models.sort()

  answer = t('model.index') + '\n'.join(models)
  await msg.reply(answer)
  msg.stop_propagation()


@Client.on_message(filters.command(["model"]))
async def get(_: Client, msg: Message) -> None:
  await msg.reply(t('model.get', {'/name/': shared.model_name}))
  msg.stop_propagation()

# @Client.on_message(filters.command(["set_model"]))
# async def put(_: Client, msg: Message) -> None:
#   if msg.command[1]+'.json' in listdir(getcwd() + '/characters'):
#     shared.character = msg.command[1]
#     status = 'succesful'
#   else:
#     status = 'failed'

#   answer = t.model['put'][status].replace('/name/', shared.character)
#   await msg.reply(answer)
  msg.stop_propagation()
