from modules import shared

from os.path import dirname
from ..src.i18n import get_i18n

root_path = dirname(dirname(__file__))
t = get_i18n()

try:
  from pyrogram import Client, filters
  from pyrogram.types import Message
except:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', {'/path/': root_path}))

@Client.on_message(filters.command(['gradio']))
async def gradio(_: Client, msg: Message) -> None:
  print(shared.gradio, flush=True)
  await msg.reply(shared.gradio)
  msg.stop_propagation()

@Client.on_message(filters.command(['settings']))
async def settings(_: Client, msg: Message) -> None:
  print(shared.settings, flush=True)
  await msg.reply(shared.settings)
  msg.stop_propagation()
