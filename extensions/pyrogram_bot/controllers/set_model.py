from os import getcwd, listdir
from os.path import join

from pyrogram.types import Message

from modules import shared, models, text_generation

def unload_model():
  shared.model = shared.tokenizer = None
  text_generation.clear_torch_cache()

def load_model(selected_model: str):
  shared.model_name = selected_model
  shared.model, shared.tokenizer = models.load_model(shared.model_name)

def set_model(selected_model: str) -> bool:
  model_path = join(getcwd(), 'models')

  if selected_model == shared.model_name: return False
  if selected_model not in listdir(model_path): return False

  unload_model()
  load_model(selected_model)
  return True

async def action(app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  status = 'succesful' if set_model(msg.command[1]) else 'failed'
  answer = app.t(f'model.put.{status}', {'/name/': shared.model_name})

  await new_msg.edit(answer)
  msg.stop_propagation()
