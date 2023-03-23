from modules import shared

from ..client import Client as app

@app.on_message(app.filters.command(['gradio']))
async def gradio(_: app, msg: app.types.Message) -> None:
  print(shared.gradio, flush=True)
  await msg.reply(shared.gradio)
  msg.stop_propagation()

@app.on_message(app.filters.command(['settings']))
async def settings(_: app, msg: app.types.Message) -> None:
  print(shared.settings, flush=True)
  await msg.reply(shared.settings)
  msg.stop_propagation()
