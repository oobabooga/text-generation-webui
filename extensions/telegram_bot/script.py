from threading import Thread
from modules.text_generation import generate_reply
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
    "token": "",
    'bot_mode': "answer", #TBC, planed "chat" mode
    'bot_context': "Bot persona: Bot is cyber-assistant who help User.\nWorld scenario: This is conversation between User and Bot.\n<START>", #context for bot, added before <START>
    'user_prefix': "\nUser: ", #Adding before user message
    'user_postfix': "", #Adding after user message
    'bot_prefix': "\nBot: ", #Adding before bot message
    'bot_welcome': {"en": "Hi! I am you cyber-assistant!",
                    "ru": "Привет! Я ваш кибер-асистент!", },  # Bot welcome message!
}


class tg_Handler():
    # =============================================================================
    # start bot
    def __init__(self, bot_mode: str, bot_context: str, user_prefix: str, user_postfix: str,
                 bot_prefix: str, bot_welcome):
        self.bot_mode = bot_mode
        self.bot_context = bot_context
        self.user_prefix = user_prefix
        self.user_postfix = user_postfix
        self.bot_prefix = bot_prefix
        self.bot_welcome = bot_welcome
        self.user_history = {}
        self.button = InlineKeyboardMarkup([[InlineKeyboardButton(text=("Reset memory bot memory (not chat)"), callback_data='Reset'),
                       InlineKeyboardButton(text=("Continue previous message"), callback_data='Continue')]])

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
            message_text = self.bot_welcome["I am bot!"]
        context.bot.send_message(chat_id=upd.effective_chat.id, text=message_text)

    def cb_get_message(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_get_message, args=(upd, context)).start()

    def tr_get_message(self, upd: Update, context: CallbackContext):
        user_text = upd.message.text
        chatId = upd.message.chat.id
        #print(chatId, "IN<", user_text)
        answer = self.generate_answer(user_text=user_text, chatId=chatId)
        #print(chatId, "OUT>")
        context.bot.send_message(chat_id=chatId, text=answer, reply_markup=self.button)

    # =============================================================================
    # button handler
    def cb_opt_button(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_opt_button, args=(upd, context)).start()

    def tr_opt_button(self, upd: Update, context: CallbackContext):
        query = upd.callback_query
        query.answer()
        chatId = query.message.chat.id
        msg_text = query.message.text
        option = query.data
        if option == "Reset":
            if chatId in self.user_history.keys():
                self.user_history[chatId] = ''
            context.bot.send_message(chat_id=chatId, text="<CONVERSATION CONTEXT DELETE, BOT LOST HIS MEMORY>")
        elif option == "Continue":
            answer = self.generate_answer(user_text='', chatId=chatId, mode='continue')
            context.bot.send_message(chat_id=chatId, text=answer, reply_markup=self.button)

    # =============================================================================
    # answer generator
    def generate_answer(self, user_text, chatId, mode='chat'):
        if chatId in self.user_history.keys():
            history = self.user_history[chatId] + "\n"
        else:
            history = ''
        if mode == "chat":
            prompt = self.bot_context + history + self.user_prefix + user_text + self.user_postfix + self.bot_prefix
        else:
            prompt = self.bot_context + history
        generator = generate_reply(
            question=prompt, stopping_strings=['\n'], max_new_tokens=256,
            do_sample=True, temperature=0.6, top_p=0.1, top_k=40, typical_p=1,
            repetition_penalty=1.1, encoder_repetition_penalty=1,
            min_length=0, no_repeat_ngram_size=0,
            num_beams=1, penalty_alpha=0, length_penalty=1.1,
            early_stopping=True, seed=-1,
        )
        answer = ''
        for a in generator:
            answer = a
        if chatId in self.user_history.keys():
            self.user_history[chatId] = self.user_history[chatId] + "\n" + answer
        else:
            self.user_history[chatId] = answer
        return answer


def run_server():
    tg_server = tg_Handler(params['bot_mode'], params['bot_context'], params['user_prefix'],
                           params['user_postfix'], params['bot_prefix'], params['bot_welcome'])
    tg_server.run_telegramm_bot(params['token'])


def setup():
    Thread(target=run_server, daemon=True).start()
