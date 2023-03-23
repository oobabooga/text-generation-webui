from modules import shared

from ..client import Client as app
from ..services.model import get_model, set_model

Message = app.types.Message
filters = app.filters

@app.on_message(app.filters.command(["characters"]))
async def index(_: app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  answer = app.t('model.index') + '\n'.join(get_model())

  await new_msg.edit(answer)
  msg.stop_propagation()

@app.on_message(filters.command(["character"]))
async def get(_: app, msg: Message) -> None:
  await msg.reply(app.t('character.get', {'/name/': shared.character}))
  msg.stop_propagation()

@app.on_message(filters.command(["set_character"]))
async def put(_: app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  status = 'succesful' if set_model(msg.command[1]) else 'failed'
  answer = app.t(f'model.put.{status}', {'/name/': shared.model_name})

  await new_msg.edit(answer)
  msg.stop_propagation()
