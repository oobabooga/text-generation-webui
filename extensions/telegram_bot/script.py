from threading import Thread
from modules.text_generation import generate_reply
from pathlib import Path
import json
from os import listdir
from os.path import exists
from copy import deepcopy
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
    "token": "TELEGRAM_TOKEN",  # Telegram bot token! Ask https://t.me/BotFather to get!
    'bot_mode': "chat",  # chat, chat-restricted, notebook
    'character_to_load': "Example.json",  # character json file from text-generation-webui/characters
}


class TelegramBotWrapper():
    # #
    users: dict = {}  # dict of User data dicts, here placed all users session info.
    default_users_data = {"name1": "You",  # user name
                          "name2": "Bot",  # bot name
                          "context": "",  # context of conversation, example: "Conversation between Bot and You"
                          "history": [],  # "history": [["Hi!", "Hi there!","Who are you?", "I am you assistant."]],
                          "msg_id": [],  # "msgid": [143, 144, 145, 146],
                          "greeting": 'Hi',  # just greeting message from bot
                          }


    def __init__(self, bot_mode="chat", char_file="Example", ):
        # Set bot paths, can be changed later
        self.history_dir_path = "extensions/telegram_bot/history"
        self.default_token_file_path = "extensions/telegram_bot/telegram_token.txt"
        self.characters_dir_path = "characters"
        # Set bot_mode and eos presets, default_char
        self.bot_mode = bot_mode
        self.char_file = char_file
        # Set buttons default list - if chat-restricted user cant change char or get help
        if self.bot_mode == "chat":
            self.button = InlineKeyboardMarkup(
                [[InlineKeyboardButton(text=("▶Continue"), callback_data='Continue'),
                  InlineKeyboardButton(text=("🔄Regenerate"), callback_data='Regen'),
                  InlineKeyboardButton(text=("✂Cutoff"), callback_data='Cutoff'),
                  InlineKeyboardButton(text=("🚫Reset memory"), callback_data='Reset'),
                  InlineKeyboardButton(text=("👨‍👩‍👧‍👦Characters"), callback_data='Chars'),
                  ]])
        if self.bot_mode == "chat-restricted":
            self.button = InlineKeyboardMarkup(
                [[InlineKeyboardButton(text=("▶Continue"), callback_data='Continue'),
                  InlineKeyboardButton(text=("🔄Regenerate"), callback_data='Regen'),
                  InlineKeyboardButton(text=("✂Cutoff"), callback_data='Cutoff'),
                  InlineKeyboardButton(text=("🚫Reset memory"), callback_data='Reset'),
                  ]])

    # =============================================================================
    # Run bot with token! Initiate updater obj!
    def run_telegram_bot(self, bot_token="", token_file=""):
        if bot_token == "":
            if token_file == "":
                token_file = self.default_token_file_path
            bot_token = open(token_file, "r", encoding='utf-8').read()
        self.updater = Updater(token=bot_token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler(['start', 'reset'], self.cb_get_command))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text, self.cb_get_message))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.cb_opt_button))
        self.updater.start_polling()
        print("Telegram bot started!", self.updater)

    # =============================================================================
    # Command handler
    def cb_get_command(self, upd: Update, context: CallbackContext):
        if upd.message.text == "/start":
            Thread(target=self.send_welcome_message, args=(upd, context)).start()

    def send_welcome_message(self, upd: Update, context: CallbackContext):
        self.init_user_or_load_history(upd.effective_chat.id)
        context.bot.send_message(chat_id=upd.effective_chat.id, text=self.users[upd.effective_chat.id]["greeting"])

    # =============================================================================
    # Additional telegram actions (history reset, char changing, welcome and others)
    def last_message_markup_clean(self, context: CallbackContext, chat_id: int):
        # delete buttons if there is user and user have at least one message id
        if chat_id in self.users.keys():
            if len(self.users[chat_id]["msg_id"]) > 0:
                try:
                    context.bot.editMessageReplyMarkup(chat_id=chat_id, message_id=self.users[chat_id]["msg_id"][-1],
                                                       reply_markup=None)
                except Exception as e:
                    print("last_message_markup_clean", e)

    def reset_history_button(self, upd: Update, context: CallbackContext):
        chat_id = upd.callback_query.message.chat.id
        if chat_id in self.users.keys():
            if len(self.users[chat_id]["msg_id"]) > 0:
                self.last_message_markup_clean(context, chat_id)
            self.users[chat_id]["history"] = []
            self.users[chat_id]["msg_id"] = []
        context.bot.send_message(chat_id=chat_id, text="<BOT LOST MEMORY!>\nSend /start or any text for new session.")

    def get_characters_json_list(self) -> list:
        file_list = listdir(self.characters_dir_path)
        char_list = []
        i = 1
        for file_name in file_list:
            if file_name[-5:] == ".json":
                i += 1
                char_list.append(file_name)
        return char_list

    def load_char_message(self, upd: Update, context: CallbackContext):
        chat_id = upd.message.chat.id
        self.last_message_markup_clean(context, chat_id)
        char_list = self.get_characters_json_list()
        char_file = char_list[int(upd.message.text.split("_LOAD_")[-1].strip().lstrip())]
        self.users[chat_id] = self.load_char_json_file(char_file=char_file)
        context.bot.send_message(chat_id=chat_id, text="<NEW CHAR LOADED>\nSend /start or any text for new session.")

    def init_user_or_load_history(self, chat_id):
        if chat_id not in self.users.keys():
            #If not exist - check history file
            if exists(f"{self.history_dir_path}/{chat_id}.json"):
                try:
                    data = open(Path(f'{self.history_dir_path}/{chat_id}.json'), 'r', encoding='utf-8').read()
                    self.users[chat_id] = json.loads(data)
                except Exception as e:
                    print("user_init", e)
                    self.users[chat_id] = self.load_char_json_file(char_file=self.char_file)
            # If no history file - load default char
            else:
                self.users[chat_id] = self.load_char_json_file(char_file=self.char_file)

    def save_user_history(self, chat_id):
        if chat_id in self.users.keys():
            with open(f"{self.history_dir_path}/{chat_id}.json", 'w', encoding='utf-8') as users_f:
                users_f.write(json.dumps(self.users[chat_id]))

    # =============================================================================
    # Text message handler
    def cb_get_message(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_get_message, args=(upd, context)).start()

    def tr_get_message(self, upd: Update, context: CallbackContext):
        # If starts with _LOAD_ - loading char!
        if upd.message.text.startswith("/_LOAD_") and self.bot_mode != "chat-restricted":
            self.load_char_message(upd, context)
            return True
        # If not char load - continue generating
        user_text = upd.message.text
        chat_id = upd.message.chat.id
        self.init_user_or_load_history(chat_id) #  if no such user - load char
        # send "typing" message, generate answer, repalce "typing" to answer
        message = context.bot.send_message(chat_id=chat_id, text=self.users[chat_id]["name2"] + " typing...")
        answer = self.generate_answer(user_in=user_text, chat_id=chat_id)
        context.bot.editMessageText(chat_id=chat_id, message_id=message.message_id, text=answer,
                                    reply_markup=self.button)
        # clear buttons on last message (if exist in current thread) and add message_id to message_history
        self.last_message_markup_clean(context, chat_id)
        self.users[chat_id]["msg_id"].append(message.message_id)
        self.save_user_history(chat_id)
        return True

    # =============================================================================
    # button handler
    def cb_opt_button(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_opt_button, args=(upd, context)).start()

    def tr_opt_button(self, upd: Update, context: CallbackContext):
        query = upd.callback_query
        query.answer()
        chat_id = query.message.chat.id
        msg_id = query.message.message_id
        msg_text = query.message.text
        option = query.data
        if chat_id not in self.users.keys():  # if no history for this message - do not answer, del buttons
            self.init_user_or_load_history(chat_id)
        if msg_id not in self.users[chat_id]["msg_id"]:  # if msg not in message history - do not answer, del buttons
            context.bot.editMessageText(msg_text + "\n<THREAD MEMORY LOST>\nSend /start or any text for new session.",
                                        chat_id, msg_id, reply_markup=None)
        elif option == "Reset":  # if no history for this message - do not answer, del buttons
            self.reset_history_button(upd=upd, context=context)
            self.save_user_history(chat_id)
        elif option == "Regen":  # Regenerate is like others generating, but delete previous bot answer
                # add pretty "retyping"
                context.bot.editMessageText(msg_text + '\n' + self.users[chat_id]["name2"] + " retyping...",
                                            chat_id, msg_id, reply_markup=self.button)
                # remove last bot answer, read and remove last user reply
                self.users[chat_id]["history"].pop()
                user_in = self.users[chat_id]["history"].pop().replace(self.users[chat_id]["name1"] + ": ", "")
                # get answer and replace message text!
                answer = self.generate_answer(user_in=user_in, chat_id=chat_id)
                context.bot.editMessageText(answer, chat_id, msg_id, reply_markup=self.button)
                self.save_user_history(chat_id)
        elif option == "Continue":  # continue is like others generating
                # add pretty "retyping" (like any other text generating)
                message = context.bot.send_message(chat_id=chat_id, text=self.users[chat_id]["name2"] + " typing...")
                answer = self.generate_answer(user_in='', chat_id=chat_id)
                context.bot.editMessageText(chat_id=chat_id, message_id=message.message_id, text=answer,
                                            reply_markup=self.button)
                self.last_message_markup_clean(context, chat_id)
                self.users[chat_id]["msg_id"].append(message.message_id)
                self.save_user_history(chat_id)
        elif option == "Cutoff":
            if chat_id in self.users.keys():
                # Edit last msg_id (stricted lines)
                context.bot.editMessageText("<s>" + self.users[chat_id]["history"][-1] + "</s>", chat_id,
                                            self.users[chat_id]["msg_id"][-1], parse_mode="HTML")
                self.users[chat_id]["history"].pop()
                self.users[chat_id]["history"].pop()
                self.users[chat_id]["msg_id"].pop()
                if len(self.users[chat_id]["msg_id"]) > 0:
                    context.bot.editMessageText(self.users[chat_id]["history"][-1], chat_id,
                                                self.users[chat_id]["msg_id"][-1], reply_markup=self.button)
                self.save_user_history(chat_id)
            else:
                context.bot.editMessageText(msg_text + "\n<HISTORY LOST>", chat_id, msg_id, reply_markup=self.button)
        elif option == "Chars":
            char_list = self.get_characters_json_list()
            to_send = []
            for i, char in enumerate(char_list):
                to_send.append("/_LOAD_" + str(i) + " " + char.replace(".json", ""))
                if i % 50 == 0 and i != 0:
                    context.bot.send_message(chat_id=chat_id, text="\n".join(to_send))
                    to_send = []
            if len(to_send) > 0:
                context.bot.send_message(chat_id=chat_id, text="\n".join(to_send))

    # =============================================================================
    # answer generator
    def generate_answer(self, user_in, chat_id):
        # If notebook - append to history only user text; if char - add "name1/2:"; if user_in empty - no user text
        if self.bot_mode == "notebook":
            self.users[chat_id]["history"].append(user_in)
        elif user_in == "":
            self.users[chat_id]["history"].append("")
            self.users[chat_id]["history"].append(self.users[chat_id]["name2"] + ":")
        else:
            self.users[chat_id]["history"].append(self.users[chat_id]["name1"] + ":" + user_in)
            self.users[chat_id]["history"].append(self.users[chat_id]["name2"] + ":")
        # Set eos_token and stopping_strings
        stopping_strings = []
        eos_token = None
        if self.bot_mode in ["chat", "chat-restricted"]:
            #dont know why, but better works without stopping_strings
            #stopping_strings = [f'\n{self.users[chat_id]["name2"]}:', f'\n{self.users[chat_id]["name1"]}:']
            eos_token = '\n'
        # Make prompt
        prompt = self.users[chat_id]["context"] + "\n" + "\n".join(self.users[chat_id]["history"])
        # Generate!
        generator = generate_reply(
            question=prompt, max_new_tokens=1024,
            do_sample=True, temperature=0.72, top_p=0.73, top_k=0, typical_p=1,
            repetition_penalty=1.1, encoder_repetition_penalty=1,
            min_length=0, no_repeat_ngram_size=0,
            num_beams=1, penalty_alpha=0, length_penalty=1,
            early_stopping=False, seed=-1,
            eos_token=eos_token, stopping_strings=stopping_strings
        )
        # This is "bad" implementation of getting answer
        answer = ''
        for a in generator:
            answer = a
        # If generation result - zero - return  "Empty answer."
        if len(answer) < 1:
            return "Empty answer"
        # add to last message in history "name2:" generated answer
        self.users[chat_id]["history"][-1] = self.users[chat_id]["history"][-1] + answer
        return answer

    # =============================================================================
    # load characters char_file.json from ./characters
    def load_char_json_file(self, char_file):
        # Copy default user data
        user = deepcopy(self.default_users_data.copy())
        # Try to read char file. If reading fail - return default user data
        try:
            data = json.loads(open(Path(f'{self.characters_dir_path}/{char_file}'), 'r', encoding='utf-8').read())
            #  load persona and scenario
            if 'you_name' in data and data['you_name'] != '':
                user["name2"] = data['char_name']
            else:
                user["name1"] = "You"
            if 'char_name' in data and data['char_name'] != '':
                user["name2"] = data['char_name']
            if 'char_persona' in data and data['char_persona'] != '':
                user["context"] += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
            if 'world_scenario' in data and data['world_scenario'] != '':
                user["context"] += f"Scenario: {data['world_scenario']}\n"
            #  add dialoque examples
            if 'example_dialogue' in data and data['example_dialogue'] != '':
                data['example_dialogue'] = data['example_dialogue'].replace('{{user}}', user["name1"])
                data['example_dialogue'] = data['example_dialogue'].replace('{{char}}', user["name2"])
                data['example_dialogue'] = data['example_dialogue'].replace('<USER>', user["name1"])
                data['example_dialogue'] = data['example_dialogue'].replace('<BOT>', user["name2"])
                user["context"] += f"{data['example_dialogue'].strip()}\n"
            #  after <START> add char greeting
            user["context"] += f"{user['context'].strip()}\n<START>\n"
            if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
                user["context"] += '\n' + data['char_greeting'].strip()
                user["greeting"] = data['char_greeting'].strip()
        except Exception as e:
            print("load_char_json_file", e)
        return user


def run_server():
    # example with char load context:
    tg_server = TelegramBotWrapper(bot_mode=params['bot_mode'], char_file=params['character_to_load'])
    tg_server.run_telegram_bot()  # by default - read in extensions/telegram_bot/telegram_token.txt


def setup():
    Thread(target=run_server, daemon=True).start()
