import asyncio
from os import getcwd
from os.path import dirname, join
from json import loads as parse

from modules import shared, text_generation

from .i18n import get_i18n

root_path = dirname(dirname(__file__))

t = get_i18n()

try:
  from pyrogram import Client
  from pyrogram.types import Message, User
except:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', {'/path/': root_path}))

parameters = {
  'max_length': 200,
  'temperature': 0.5,
  'top_p': 1,
  'typical_p': 1,
  'repetition_penalty': 1.1,
  'top_k': 0,
}

def prepare_message_text(msg: Message) -> str:
  if msg.text:    return msg.text
  if msg.caption: return msg.caption
  return ''

def prepare_message_sender(msg: Message, bot: User) -> str:
  if msg.from_user:
    if msg.from_user.id == bot.id:
      return shared.settings['name2_pygmalion']
    return 'You'
  return ''

def prepare_telegram_message(msg: Message, bot: User) -> str:
  author = prepare_message_sender(msg, bot)
  text = prepare_message_text(msg)

  if not msg.command and author:
    return f'{author}: {text}'

  return ''

async def prepare_message_history(app: Client, msg: Message) -> str:
  ids = [i for i in range(msg.id, msg.id-10, -1)]
  messages, bot = await asyncio.gather(*[
    app.get_messages(msg.chat.id, ids),
    app.get_me()
  ])
  messages = [prepare_telegram_message(message, bot) for message in messages]
  messages = [x for x in messages if x]
  question = '\n'.join(messages)

  return question

def replace_all(string: str, replaceable: dict[str, str]) -> str:
  for before, after in replaceable.items():
    string = string.replace(before, after)
  return string

def prepare_char_persona() -> str:
  char = {}
  char_path = join(getcwd(), 'characters', f'{shared.character}.json')
  with open(char_path, 'r') as char_file:
    char_json = char_file.read()
    char = parse(char_json)

  shared.settings['name2_pygmalion'] = char['char_name']

  dialogues = replace_all(char['example_dialogue'], {
    '{{user}}': 'You',
    '{{char}}': char['char_name'],
  })
  prompt = (
    f"{char['char_name']}'s Persona: {char['char_persona']}\n"
    f"Scenario: {char['world_scenario']}\n\n"
    f"{dialogues}"
  )

  return prompt

def prepare_answer(answer: str, question: str) -> str:
  answer = answer.replace(question, '')
  if answer == '':        return 'empty'
  if len(answer) >= 4095: return answer[-4095:]

  return answer

def generate_bot_reply(question: str) -> str:
  answer = ''
  for a in text_generation.generate_reply(
    question = question,
    max_new_tokens  = parameters.get('max_length', 200),
    temperature = parameters.get('temperature', 0.5),
    top_p = parameters.get('top_p', 1),
    typical_p = parameters.get('typical_p', 1),
    repetition_penalty = parameters.get('repetition_penalty', 1.1),
    top_k = parameters.get('top_k', 0),
    encoder_repetition_penalty = 1,
    do_sample = True,
    min_length = 0,
    no_repeat_ngram_size = 0,
    num_beams = 1,
    penalty_alpha = 0,
    length_penalty = 1,
    early_stopping = False,
  ):
    print(a, flush=True)
    answer = a

  return answer

def prepare_ai_prompt(persona:str, message_history: str) -> str:
  return (
    f"{persona}\n"
    f"{message_history}\n"
    f"{shared.settings['name2_pygmalion']}\n"
  )
