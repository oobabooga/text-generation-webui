from pyrogram.types import Message

from modules import shared

async def gradio(_, msg: Message) -> None:
  print(shared.gradio, flush=True)
  await msg.reply(shared.gradio)
  msg.stop_propagation()

async def settings(_, msg: Message) -> None:
  print(shared.settings, flush=True)
  await msg.reply(shared.settings)
  msg.stop_propagation()

async def info(app, msg: Message) -> None:
  t = app.t
  answer = ''
  answer_dict = {
    t('model.is'): shared.model_name,
    t('character.is'): shared.character,
    t('lang.is'): t(f'lang.{app.lang}'),
  }

  for key, value in answer_dict.items():
    answer += f"{key}: {value}\n"

  await msg.reply(answer)
  msg.stop_propagation()
