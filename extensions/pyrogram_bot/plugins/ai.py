import os

from modules.text_generation import generate_reply

from ..src.i18n import get_i18n

path = os.path.dirname(__file__)

t = get_i18n()

parameters = {
  'max_length': 200,
  'temperature': 0.5,
  'top_p': 1,
  'typical_p': 1,
  'repetition_penalty': 1.1,
  'top_k': 0,
}

try:
  from pyrogram import Client, filters
  from pyrogram.types import Message
except:
  raise Exception(t['error']['dependencies_not_installed'].replace('/path/', path))

def message_text(msg: Message) -> str:
  if msg.text: return msg.text
  if msg.caption: return msg.caption
  return ''

def message_sender(msg: Message) -> str:
  if msg.sender_chat:
    return msg.sender_chat.title

  if msg.from_user:
    user = msg.from_user

    if user.username:  
      return user.username
      
    return user.first_name
  
  return ''
    
def format_message(msg: Message) -> str:
  author = message_sender(msg)
  text = message_text(msg)

  if author:
    return f'{author}: {text}'

  return ''

async def format_question(app: Client, msg: Message) -> str:
  ids = [i for i in range(msg.id, msg.id-10, -1)]
  messages = await app.get_messages(msg.chat.id, ids)
  messages = list(map(format_message, messages))
  messages = [x for x in messages if x]
  question = '\n'.join(messages)

  return question

def format_answer(answer: str) -> str:
  if answer == '':        return 'empty'
  if len(answer) >= 4095: return answer[-4095:]

  return answer

def generate_bot_reply(question: str) -> str:
  generator = generate_reply(
    question=question, 
    max_new_tokens = parameters.get('max_length', 200), 
    do_sample=True, 
    temperature=parameters.get('temperature', 0.5), 
    top_p=parameters.get('top_p', 1), 
    typical_p=parameters.get('typical_p', 1), 
    repetition_penalty=parameters.get('repetition_penalty', 1.1), 
    encoder_repetition_penalty=1, 
    top_k=parameters.get('top_k', 0), 
    min_length=0, 
    no_repeat_ngram_size=0, 
    num_beams=1, 
    penalty_alpha=0, 
    length_penalty=1,
    early_stopping=False,
  )

  answer = ''
  for a in generator:
    answer = a
  
  return answer

@Client.on_message(filters.user(600432868), group=999)
async def send_reply(app: Client, msg: Message) -> None:
  question = await format_question(app, msg)
  answer = generate_bot_reply(question)
  answer = format_answer(answer)

  await msg.reply(answer)