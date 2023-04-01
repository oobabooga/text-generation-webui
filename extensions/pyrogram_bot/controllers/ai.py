from typing import Generator

from pyrogram.types import Message

from modules import chat, shared

parameters = {
  'max_length': 64,
  'temperature': 0.5,
  'top_p': 1,
  'typical_p': 1,
  'repetition_penalty': 1.3,
  'top_k': 0,
}

def prepare_message_text(msg: Message) -> str:
  if msg.text:    return msg.text
  if msg.caption: return msg.caption
  return ''

def history_last() -> str:
  try:
    return shared.history['visible'][-1][-1]
  except IndexError:
    return ''

def generate_bot_promt(msg: Message) -> str:
  return chat.generate_chat_prompt(
    user_input = prepare_message_text(msg),
    max_new_tokens = parameters.get('max_length', 200),
    name1 = shared.settings['name1'],
    name2 = shared.settings['name2'],
    context = shared.settings['context'],
    chat_prompt_size = 2048,
  )

def generate_bot_reply(text: str) -> Generator[list, None, list | None]:
  return chat.chatbot_wrapper(
    text = text,
    max_new_tokens = parameters.get('max_length', 200),
    do_sample = True,
    temperature = parameters.get('temperature', 0.5),
    top_p = parameters.get('top_p', 1),
    typical_p = parameters.get('typical_p', 1),
    repetition_penalty = parameters.get('repetition_penalty', 1.1),
    encoder_repetition_penalty = 1,
    top_k = parameters.get('top_k', 0),
    min_length = 0,
    no_repeat_ngram_size = 0,
    num_beams = 1,
    penalty_alpha = 0,
    length_penalty = 1,
    early_stopping = False,
    seed = -1,
    name1 = shared.settings['name1'],
    name2 = shared.settings['name2'],
    context = shared.settings['context'],
    check = None,
    chat_prompt_size = 2048,
    chat_generation_attempts=1,
    regenerate=False
  )

async def action(app, msg: Message) -> None:
  new_msg = await msg.reply(app.t('message.is_typing'))

  promt = generate_bot_promt(msg)
  answer_generator = generate_bot_reply(promt)

  for _ in answer_generator:
    if history_last()[-1] in ['.', '!', '?']:
      text = f"{shared.history['visible'][-1][-1]} { app.t('message.is_typing')}"
      new_msg = await new_msg.edit(text)

  if not history_last() == new_msg.text:
    await new_msg.edit(history_last())

  msg.stop_propagation()
