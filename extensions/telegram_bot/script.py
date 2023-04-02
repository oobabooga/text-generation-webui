from threading import Thread
from modules.text_generation import generate_reply
from pathlib import Path
import json
from telegram import Update
from telegram import InlineKeyboardButton
from telegram import InlineKeyboardMarkup
from telegram.ext import CallbackContext
from telegram.ext import Filters
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import CallbackQueryHandler
from telegram.ext import Updater

params = {
    "token": "PLACE_TELEGRAM_TOKEN_HERE", # Telegram bot token! Ask https://t.me/BotFather to get!
    'character_to_load': "Example", #"chat" or "notebook"
    'bot_mode': "chat", #"chat" or "notebook"
    'bot_welcome': {"en": "Hi! I am you cyber-assistant!",
                    "ru": "Привет! Я ваш кибер-асистент!", },  # Bot welcome message!
}

char = {
    'context': """Answerer's Persona: Answerer is girl whos primary target is find answers. Answerer like to learn something new nad give correctly answers.\nScenario: This is a convercation between Answerer and You in real world.\n<START>""",
    'name1': "You",
    'name2': "Answerer",
}


class TelegramBotWrapper():
    # init
    def __init__(self, bot_mode="chat", bot_context="", name2="Bot",
                 name1="You", **kwargs):
        # Set chat context and names, raw style
        self.bot_context = bot_context
        self.name1 = name1
        self.name2 = name2
        # Set bot_mode
        self.bot_mode = bot_mode
        if self.bot_mode == "chat":
            self.stoping_strings = [f"\n{self.name2}:", f"\n{self.name1}:"]
            self.eos_token = '\n'
        elif self.bot_mode == "notebook":
            self.stoping_strings = []
            self.eos_token = None
        else:
            self.stoping_strings = []
            self.eos_token = None
        # Set welcome
        if "bot_welcome" in kwargs:
            self.bot_welcome = kwargs["bot_welcome"]
        else:
            self.bot_welcome = {"en": "Hi!", "ru": "Привет!", }
        self.user_history = {}
        # Set buttoms default list
        self.button = InlineKeyboardMarkup(
            [[InlineKeyboardButton(text=("▶Continue"), callback_data='Continue'),
              InlineKeyboardButton(text=("✂Cut off mem"), callback_data='Cutoff'),
              InlineKeyboardButton(text=("❌Cut+Delete"), callback_data='CutDel'),
              InlineKeyboardButton(text=("🚫Reset memory"), callback_data='Reset'),
              ]])
        # Set load char char_file exist, overwrite raw style
        if "char_file" in kwargs:
            self.load_char_from_file(kwargs["char_file"])

    # =============================================================================
    # run bot
    def run_telegramm_bot(self, bot_token: str):
        self.updater = Updater(token=bot_token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler('start', self.send_welcome_message))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text, self.cb_get_message))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.cb_opt_button))
        self.updater.start_polling()
        print("Telegramm bot started!", self.updater)

    # =============================================================================
    # Text message handler
    def send_welcome_message(self, upd: Update, context: CallbackContext):
        if upd.message.from_user.language_code in self.bot_welcome.keys():
            message_text = self.bot_welcome[upd.message.from_user.language_code]
        else:
            message_text = self.bot_welcome['en']
        context.bot.send_message(chat_id=upd.effective_chat.id, text=message_text)

    def cb_get_message(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_get_message, args=(upd, context)).start()

    def tr_get_message(self, upd: Update, context: CallbackContext):
        user_text = upd.message.text
        chatId = upd.message.chat.id
        message = context.bot.send_message(chat_id=chatId, text=self.name2 + " typing...")
        answer = self.generate_answer(user_text=user_text, chatId=chatId)
        context.bot.editMessageText(chat_id=chatId, message_id=message.message_id, text=answer, reply_markup=self.button)
        #context.bot.send_message(chat_id=chatId, text=answer, reply_markup=self.button)

    # =============================================================================
    # button handler
    def cb_opt_button(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_opt_button, args=(upd, context)).start()

    def tr_opt_button(self, upd: Update, context: CallbackContext):
        query = upd.callback_query
        query.answer()
        chatId = query.message.chat.id
        msg_id = query.message.message_id
        msg_text = query.message.text
        option = query.data
        if option == "Reset":
            if chatId in self.user_history.keys():
                self.user_history[chatId] = ''
            context.bot.send_message(chat_id=chatId, text="<CONVERSATION CONTEXT DELETE, BOT LOST HIS MEMORY>\n/start")
        elif option == "Cutoff":
            if chatId in self.user_history.keys():
                self.user_history[chatId] = self.user_history[chatId].replace(self.name2 + ":" + msg_text, "")
            context.bot.editMessageText("✂" + msg_text + "✂", chatId, msg_id, reply_markup=self.button)
        elif option == "CutDel":
            if chatId in self.user_history.keys():
                self.user_history[chatId] = self.user_history[chatId].replace(self.name2 + ":" + msg_text, "")
            context.bot.delete_message(chatId, msg_id)
        elif option == "Continue":
            message = context.bot.send_message(chat_id=chatId, text=self.name2 + " typing...")
            answer = self.generate_answer(user_text='continue', chatId=chatId)
            context.bot.editMessageText(chat_id=chatId, message_id=message.message_id, text=answer,
                                        reply_markup=self.button)

    # =============================================================================
    # answer generator
    def generate_answer(self, user_text, chatId):
        if chatId not in self.user_history.keys():
            self.user_history[chatId] = ''
        if user_text != "":
            if self.bot_mode == "notebook":
                self.user_history[chatId] = self.user_history[chatId] + "\n" + user_text
            else:
                self.user_history[chatId] = self.user_history[chatId] + "\n" + self.name1 + ":" + user_text + "\n" + self.name2 + ":"
        prompt = self.bot_context + "\n" + self.user_history[chatId]
        generator = generate_reply(
            question=prompt, max_new_tokens=1024,
            do_sample=True, temperature=0.72, top_p=0.73, top_k=0, typical_p=1,
            repetition_penalty=1.1, encoder_repetition_penalty=1,
            min_length=0, no_repeat_ngram_size=0,
            num_beams=1, penalty_alpha=0, length_penalty=1,
            early_stopping=True, seed=-1,
            eos_token=self.eos_token, stopping_strings=self.stoping_strings
        )
        answer = ''
        for a in generator:
            answer = a
        self.user_history[chatId] = self.user_history[chatId] + answer
        if answer in ["", " ", "\n", "\n ", " \n"]:
            answer = "Empty answer."
        return answer

    # =============================================================================
    # load characters file.json from ./characters
    def load_char_from_file(self, char_file):
        data = json.loads(open(Path(f'characters/{char_file}.json'), 'r', encoding='utf-8').read())
        self.bot_context = ''
        self.name2 = data['char_name']
        if 'char_persona' in data and data['char_persona'] != '':
            self.bot_context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
        if 'world_scenario' in data and data['world_scenario'] != '':
            self.bot_context += f"Scenario: {data['world_scenario']}\n"
        self.bot_context += f"{self.bot_context.strip()}\n<START>\n"
        if 'example_dialogue' in data and data['example_dialogue'] != '':
            data['example_dialogue'] = \
                data['example_dialogue'].replace('{{user}}', self.name1).replace('{{char}}', self.name2)
            data['example_dialogue'] = \
                data['example_dialogue'].replace('<USER>', self.name1).replace('<BOT>', self.name2)
            self.bot_context += f"{data['example_dialogue'].strip()}\n"
        if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
            self.bot_welcome = {'en': data['char_greeting']}
        else:
            self.bot_welcome = {'en': 'Conversation with ' + self.name2}


def run_server():
    # example with raw context. :
    # tg_server = TelegramBotWrapper(bot_mode=params['bot_mode'], name1=char['name1'], name2=char['name2'], context=char['context'], bot_welcome=params['bot_welcome'])
    # example with char load context:
    tg_server = TelegramBotWrapper(bot_mode=params['bot_mode'], bot_welcome=params['bot_welcome'], char_file=params['character_to_load'])
    tg_server.run_telegramm_bot(params['token'])


def setup():
    Thread(target=run_server, daemon=True).start()
