from os import environ as env
from ..client import Client as app
from ..services.ai import (
  prepare_char_persona,
  prepare_message_history,
  prepare_ai_prompt,
  generate_bot_reply,
  prepare_answer
)

@app.on_message(app.filters.chat(app.allowed_chat), group=999)
async def send_reply(app: app, msg: app.types.Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  char_persona = prepare_char_persona()
  message_history = await prepare_message_history(app, msg)
  ai_prompt = prepare_ai_prompt(char_persona, message_history)
  ai_answer = generate_bot_reply(ai_prompt)
  answer = prepare_answer(ai_answer, ai_prompt)

  await new_msg.edit(answer)
  msg.stop_propagation()
