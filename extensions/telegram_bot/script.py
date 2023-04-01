from threading import Thread
from modules.text_generation import generate_reply
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import Filters
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import Updater

params = {
    "token": "",
    'bot_mode': "answer", #TBC, planed "chat" mode
    'bot_context': "Bot persona: Bot is cyber-assistant who help User.\nWorld scenario: This is conversation between User and Bot.\n<START>", #context for bot, added before <START>
    'user_prefix': "\nUser message: ", #Adding before user message
    'user_postfix': "", #Adding after user message
    'bot_prefix': "\nBot answer: ", #Adding before bot message
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

    def run_telegramm_bot(self, bot_token: str):
        self.updater = Updater(token=bot_token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler('start', self.send_welcome_message))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text, self.cb_get_message))
        self.updater.start_polling()
        print("Telegramm bot started!", self.updater)


    # =============================================================================
    # Text message handler
    def send_welcome_message(self, upd: Update, context: CallbackContext):
        if upd.message.from_user.language_code in self.bot_welcome.keys:
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
        answer = self.generate_answer(user_text)
        #print(chatId, "OUT>")
        context.bot.send_message(chat_id=chatId, text=answer)

    def generate_answer(self, user_text):
        prompt = self.bot_context + self.user_prefix + user_text + self.user_postfix + self.bot_prefix
        generator = generate_reply(
            question=prompt, max_new_tokens=256,
            do_sample=True, temperature=0.6, top_p=0.1, top_k=40, typical_p=1,
            repetition_penalty=1.1, encoder_repetition_penalty=1,
            min_length=0, no_repeat_ngram_size=0,
            num_beams=1, penalty_alpha=0, length_penalty=1.1,
            early_stopping=True, seed=-1,
        )
        answer = ''
        for a in generator:
            answer = a
        return answer


def run_server():
    tg_server = tg_Handler(params['bot_mode'], params['bot_context'], params['user_prefix'],
                           params['user_postfix'], params['bot_prefix'], params['bot_welcome'])
    tg_server.run_telegramm_bot(params['token'])


def setup():
    Thread(target=run_server, daemon=True).start()
