import json

from pyrogram.types import Message

from modules import shared

pre = lambda text: f"```\n{text}\n```"

async def gradio(_, msg: Message) -> None:
  print(shared.gradio, flush=True)
  await msg.reply(pre(str(shared.gradio)[-4000:]))
  msg.stop_propagation()

async def settings(_, msg: Message) -> None:
  text = json.dumps(shared.settings, indent=2)
  print(text, flush=True)
  await msg.reply(pre(text))
  msg.stop_propagation()

async def history(_, msg: Message) -> None:
  text = json.dumps(shared.history['visible'], indent=2)
  print(text, flush=True)
  await msg.reply(pre(text))
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

  await msg.reply(pre(answer))
  msg.stop_propagation()
