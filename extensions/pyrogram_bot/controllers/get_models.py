from os import getcwd, listdir
from os.path import join, isdir

from pyrogram.types import Message

from modules import shared

def prepare_message() -> list[str]:
  model_path = join(getcwd(), 'models')

  models = []

  for model in listdir(model_path):
    if not isdir(join(model_path, model)):
      continue

    if model == shared.model_name:
      model = f"➢ **{model}**"

    models.append(model)

  models.sort()

  return models

async def action(app, msg: Message) -> None:
  answer = app.t('model.index') + '\n'.join(prepare_message())
  await msg.reply(answer)
  msg.stop_propagation()
