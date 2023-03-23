from modules import shared

from ..client import Client as app
from ..services.model import get_models

Message = app.types.Message
filters = app.filters

@app.on_message(filters.command(["models"]))
async def index(_: app, msg: Message) -> None:
  answer = app.t('model.index') + '\n'.join(get_models())
  await msg.reply(answer)
  msg.stop_propagation()

@app.on_message(filters.command(["model"]))
async def get(_: app, msg: Message) -> None:
  await msg.reply(app.t('model.get', { '/name/': shared.model_name }))
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
#  msg.stop_propagation()
