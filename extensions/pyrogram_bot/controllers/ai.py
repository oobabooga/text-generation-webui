from pyrogram.types import Message

from ..services.ai import generate_bot_reply, bot_reply

async def action(app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  generate_bot_reply(msg)

  await new_msg.edit(bot_reply())
  msg.stop_propagation()
