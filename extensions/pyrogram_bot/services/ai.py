from os.path import dirname

from pyrogram.types import Message

from modules import chat, shared

root_path = dirname(dirname(__file__))

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

def generate_bot_reply(msg: Message) -> str:
  promt = chat.generate_chat_prompt(
    user_input = prepare_message_text(msg),
    max_new_tokens = parameters.get('max_length', 200),
    name1 = shared.settings['name1'],
    name2 = shared.settings['name2'],
    context = shared.settings['context'],
    chat_prompt_size = 2048,
  )

  answer_generator = chat.chatbot_wrapper(
    text = promt,
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

  for _ in answer_generator:
    pass

def bot_reply() -> str:
  print(shared.history['visible'], flush)
  return shared.history['visible'][-1][-1]
