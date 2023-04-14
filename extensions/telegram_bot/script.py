from threading import Thread
from threading import Lock
from time import sleep
from modules.text_generation import generate_reply
from pathlib import Path
import json
import yaml
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
    'character_to_load': "Example.yaml",  # character json file from text-generation-webui/characters
}


class TelegramBotWrapper:
    # Default error messages
    GENERATOR_FAIL = "<GENERATION FAIL>"
    GENERATOR_EMPTY_ANSWER = "<EMPTY ANSWER>"
    UNKNOWN_TEMPLATE = "<UNKNOWN TEMPLATE>"
    # Supplementary structure
    users: dict = {}  # dict of User data dicts, here placed all users' session info.
    # Internal, changeable settings
    impersonate_prefix = "#"  # Prefix for "impersonate" messages during chatting
    default_users_data = {  # data template for user. if no default char or default char file - use this as main.
        "name1": "You",  # username
        "name2": "Bot",  # bot name
        "context": "",  # context of conversation, example: "Conversation between Bot and You"
        "user_in": [],  # "user input history": [["Hi!","Who are you?"]], need for regenerate option
        "history": [],  # "history": [["Hi!", "Hi there!","Who are you?", "I am you assistant."]],
        "msg_id": [],  # "msg_id": [143, 144, 145, 146],
        "greeting": 'Hi',  # just greeting message from bot
    }
    default_messages_template = {  # dict of messages templates for various situations. Use _VAR_ replacement
        "lost": "\n<MEMORY LOST!>\nSend /start or any text for new session.",  # When button refers to non-existing data
        "retyping": "<i>\n_NAME2_ retyping...</i>",  # added when "regenerate button" working
        "typing": "<i>\n_NAME2_ typing...</i>",  # added when generating working
        "load_char": "<CHARACTER _NAME2_ LOADED!>\n_GREETING_.",  # When new char loaded
        "reset": "<MEMORY RESET!>\nSend /start or any text for new session.",  # When history cleared
        "start": "<CHARACTER _NAME2_ LOADED>\nSend /start or message.",  # New conversation started (not used now)
    }
    generation_params = {
        'max_new_tokens': 200,
        'seed': -1.0,
        'temperature': 0.72,
        'top_p': 0.73,
        'top_k': 0,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'encoder_repetition_penalty': 1,
        'no_repeat_ngram_size': 0,
        'min_length': 0,
        'do_sample': True,
        'penalty_alpha': 0,
        'num_beams': 1,
        'length_penalty': 1,
        'early_stopping': False,
        'add_bos_token': True,
        'ban_eos_token': False,
        'truncation_length': 1024,
        'custom_stopping_strings': [],
        'end_of_turn': '',
        'chat_prompt_size': 1024,
        'chat_generation_attempts': 1,
        'stop_at_newline': False,
    }

    def __init__(self,
                 bot_mode="chat",  # bot mode - chat, chat-restricted, notebook
                 characters_dir_path="characters",  # there stored characters files
                 default_char_json="Example.json",  # name of default char.json file
                 history_dir_path="extensions/telegram_bot/history",  # there stored users history
                 default_token_file_path="extensions/telegram_bot/telegram_token.txt",  # there stored tg token
                 ):
        """
        Init telegram bot class. Use run_telegram_bot() to initiate bot.
        :param bot_mode: bot mode (chat, chat-restricted, notebook). Default is "chat".
        :param characters_dir_path: place where stored characters .json files. Default is "chat".
        :param default_char_json: name of default character.json file. Default is "chat".
        :param history_dir_path: place where stored chat history. Default is "extensions/telegram_bot/history".
        :param default_token_file_path: path to token file. Default is "extensions/telegram_bot/telegram_token.txt".
        :return: None
        """
        # Set paths to history, default token file, characters dir
        self.history_dir_path = history_dir_path
        self.default_token_file_path = default_token_file_path
        self.characters_dir_path = characters_dir_path
        # Set bot_mode and eos presets, default character json file
        self.bot_mode = bot_mode
        self.default_char_json = default_char_json
        self.load_cmd = "load"
        # Set buttons default list - if chat-restricted user can't change char or get help.
        if self.bot_mode == "chat":  # chat option
            self.button = InlineKeyboardMarkup(
                [[InlineKeyboardButton(text="▶Continue", callback_data='Continue'),
                  InlineKeyboardButton(text="🔄Regenerate", callback_data='Regen'),
                  InlineKeyboardButton(text="✂Cutoff", callback_data='Cutoff'),
                  InlineKeyboardButton(text="🚫Reset", callback_data='Reset'),
                  InlineKeyboardButton(text="🎭Chars", callback_data='Chars'),
                  ]])
            self.button_start = None
        if self.bot_mode == "chat-restricted":  # you can't change char in restricted chat mode
            self.button = InlineKeyboardMarkup(
                [[InlineKeyboardButton(text="▶Continue", callback_data='Continue'),
                  InlineKeyboardButton(text="🔄Regenerate", callback_data='Regen'),
                  InlineKeyboardButton(text="✂Cutoff", callback_data='Cutoff'),
                  InlineKeyboardButton(text="🚫Reset memory", callback_data='Reset'),
                  ]])
            self.button_start = None
        if self.bot_mode == "notebook":  # in notebook mode buttons restricted much more
            self.button = InlineKeyboardMarkup(
                [[InlineKeyboardButton(text="▶Continue", callback_data='Continue'),
                  InlineKeyboardButton(text="🚫Reset memory", callback_data='Reset'),
                  ]])
            self.button_start = None
        # dummy obj for telegram updater
        self.updater = None
        # define generator lock to prevent GPU overloading
        self.generator_lock = Lock()

    # =============================================================================
    # Run bot with token! Initiate updater obj!
    def run_telegram_bot(self, bot_token="", token_file_name=""):
        """
        This method starts telegram bot. Overwrite "self.updater" if exist
        :param bot_token: if bot_token given - use it to initiate
        :param token_file_name: if token not given - read token from this file and use it to initiate
        :return: None
        """
        if bot_token == "":
            token_file_name = self.default_token_file_path if token_file_name == "" else token_file_name
            with open(token_file_name, "r", encoding='utf-8') as token_file:
                bot_token = token_file.read()
        self.updater = Updater(token=bot_token, use_context=True)
        self.updater.dispatcher.add_handler(CommandHandler(['start', 'reset'], self.cb_get_command))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.text, self.cb_get_message))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.cb_opt_button))
        self.updater.start_polling()
        print("Telegram bot started!", self.updater)

    # =============================================================================
    # Command handlers
    def cb_get_command(self, upd: Update, context: CallbackContext):
        if upd.message.text == "/start":
            Thread(target=self.send_welcome_message, args=(upd, context)).start()

    def send_welcome_message(self, upd: Update, context: CallbackContext):
        chat_id = upd.effective_chat.id
        self.init_user(chat_id)
        send_text = self.def_msg("load_char", chat_id)
        context.bot.send_message(text=send_text, chat_id=chat_id, reply_markup=self.button_start)

    # =============================================================================
    # Additional telegram actions
    def last_message_markup_clean(self, context: CallbackContext, chat_id: int):
        if chat_id in self.users:
            # delete buttons if user exist and have at least one message_id in "msg_id"
            if len(self.users[chat_id]["msg_id"]) > 0:
                try:
                    last_msg = self.users[chat_id]["msg_id"][-1]
                    context.bot.editMessageReplyMarkup(chat_id=chat_id, message_id=last_msg, reply_markup=None)
                except Exception as e:
                    print("last_message_markup_clean", e)

    def def_msg(self, request: str, chat_id: int):
        # made default message from default_messages_template or return UNKNOWN_TEMPLATE
        if request in self.default_messages_template and chat_id in self.users:
            msg = self.default_messages_template[request]
            msg = msg.replace("_CHAT_ID_", str(chat_id))
            msg = msg.replace("_NAME1_", self.users[chat_id]["name1"])
            msg = msg.replace("_NAME2_", self.users[chat_id]["name2"])
            msg = msg.replace("_CONTEXT_", self.users[chat_id]["context"])
            msg = msg.replace("_GREETING_", self.users[chat_id]["greeting"])
            return msg
        else:
            return self.UNKNOWN_TEMPLATE

    # =============================================================================
    # Work with history!
    def reset_history_button(self, upd: Update, context: CallbackContext):
        chat_id = upd.callback_query.message.chat.id
        if chat_id in self.users:
            if len(self.users[chat_id]["msg_id"]) > 0:
                self.last_message_markup_clean(context, chat_id)
            self.users[chat_id]["history"] = []
            self.users[chat_id]["user_in"] = []
            self.users[chat_id]["msg_id"] = []
        send_text = self.def_msg("reset", chat_id)
        context.bot.send_message(text=send_text, chat_id=chat_id)

    def get_characters_files_list(self) -> list:
        file_list = listdir(self.characters_dir_path)
        char_list = []
        i = 1
        for file_name in file_list:
            if file_name.split(".")[-1] in ["json", "yaml", ".yml"]:
                i += 1
                char_list.append(file_name)
        return char_list

    def load_char_message(self, upd: Update, context: CallbackContext):
        chat_id = upd.message.chat.id
        self.last_message_markup_clean(context, chat_id)
        char_list = self.get_characters_files_list()
        char_file = char_list[int(upd.message.text.split(self.load_cmd)[-1].strip().lstrip())]
        self.users[chat_id] = self.load_char_file(char_file=char_file)
        if exists(f'{self.history_dir_path}/{chat_id}{self.users[chat_id]["name2"]}.json'):
            self.load_user_history(chat_id, self.users[chat_id]["name2"])
        send_text = self.def_msg("load_char", chat_id)
        context.bot.send_message(text=send_text, chat_id=chat_id)

    def init_user(self, chat_id):
        if chat_id not in self.users:
            # If user not exist - first load default char
            self.users[chat_id] = self.load_char_file(char_file=self.default_char_json)
            # If exist default history file (ID.json) - load it and overwrite default char
            if exists(f'{self.history_dir_path}/{chat_id}.json'):
                self.load_user_history(chat_id)
            else:
                # If no default history file, try to load default char history file (ID-CHARMANE.json)
                if exists(f'{self.history_dir_path}/{chat_id}{self.users[chat_id]["name2"]}.json'):
                    self.load_user_history(chat_id, self.users[chat_id]["name2"])

    def load_user_history(self, chat_id, name2=""):
        try:
            if exists(f'{self.history_dir_path}/{chat_id}{name2}.json'):
                user_file_path = Path(f'{self.history_dir_path}/{chat_id}{name2}.json')
                with open(user_file_path, 'r', encoding='utf-8') as user_file:
                    data = user_file.read()
                self.users[chat_id] = json.loads(data)
        except Exception as e:
            print("user_load", e)

    def save_user_history(self, chat_id, chat_name=""):
        if chat_id in self.users:
            # Save ID-CHARMANE.json file, separated history for various characters.
            user_file_path = Path(f'{self.history_dir_path}/{chat_id}{chat_name}.json')
            with open(user_file_path, 'w', encoding='utf-8') as user_file:
                user_file.write(json.dumps(self.users[chat_id]))
            # Save ID.json file, current history. File for default loading.
            default_user_file_path = Path(f'{self.history_dir_path}/{chat_id}.json')
            with open(default_user_file_path, 'w', encoding='utf-8') as user_file:
                user_file.write(json.dumps(self.users[chat_id]))

    # =============================================================================
    # Text message handler
    def cb_get_message(self, upd: Update, context: CallbackContext):
        Thread(target=self.tr_get_message, args=(upd, context)).start()

    def tr_get_message(self, upd: Update, context: CallbackContext):
        # If starts with /load_cmd - loading char!
        if upd.message.text.startswith("/" + self.load_cmd) and self.bot_mode != "chat-restricted":
            self.load_char_message(upd, context)
            return True
        # If message is not starts char loading - continue generating
        user_text = upd.message.text
        chat_id = upd.message.chat.id
        self.init_user(chat_id)  # if no such user - load char
        # send "typing" message, generate answer, replace "typing" to answer
        send_text = self.def_msg("typing", chat_id)
        message = context.bot.send_message(text=send_text, chat_id=chat_id, parse_mode="HTML")
        last_msg = message.message_id
        answer = self.generate_answer(user_in=user_text, chat_id=chat_id)
        context.bot.editMessageText(text=answer, chat_id=chat_id, message_id=last_msg, reply_markup=self.button)
        # clear buttons on last message (if exist in current thread) and add message_id to message_history
        self.last_message_markup_clean(context, chat_id)
        self.users[chat_id]["msg_id"].append(last_msg)
        self.save_user_history(chat_id, self.users[chat_id]["name2"])
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
        if chat_id not in self.users:  # if no history for this message - do not answer, del buttons
            self.init_user(chat_id)
        if msg_id not in self.users[chat_id]["msg_id"]:  # if msg not in message history - do not answer, del buttons
            send_text = msg_text + self.def_msg("lost", chat_id)
            context.bot.editMessageText(text=send_text, chat_id=chat_id, message_id=msg_id, reply_markup=None)
        elif option == "Reset":  # if no history for this message - do not answer, del buttons
            self.reset_history_button(upd=upd, context=context)
            self.save_user_history(chat_id, self.users[chat_id]["name2"])
        elif option == "Regen":  # Regenerate is like others generating, but delete previous bot answer
            # add pretty "retyping" to message text
            send_text = msg_text + self.def_msg("retyping", chat_id)
            context.bot.editMessageText(text=send_text, chat_id=chat_id, message_id=msg_id, parse_mode="HTML")
            # remove last bot answer, read and remove last user reply
            self.users[chat_id]["history"].pop()
            self.users[chat_id]["history"].pop()
            user_in = self.users[chat_id]["user_in"].pop()
            # get answer and replace message text!
            answer = self.generate_answer(user_in=user_in, chat_id=chat_id)
            print(user_in)
            context.bot.editMessageText(text=answer, chat_id=chat_id, message_id=msg_id, reply_markup=self.button)
            self.save_user_history(chat_id, self.users[chat_id]["name2"])
        elif option == "Continue":  # continue previous message is like others generating, but with "" user message
            # send "typing"
            send_text = self.def_msg("typing", chat_id)
            message = context.bot.send_message(text=send_text, chat_id=chat_id, parse_mode="HTML")
            last_msg = message.message_id
            # get answer and replace message text!
            answer = self.generate_answer(user_in='', chat_id=chat_id)
            context.bot.editMessageText(text=answer, chat_id=chat_id, message_id=last_msg, reply_markup=self.button)
            self.last_message_markup_clean(context, chat_id)
            self.users[chat_id]["msg_id"].append(message.message_id)
            self.save_user_history(chat_id, self.users[chat_id]["name2"])
        elif option == "Cutoff":
            if chat_id in self.users:
                # Edit last msg_id (strict lines)
                send_text = "<s>" + self.users[chat_id]["history"][-1] + "</s>"
                last_msg = self.users[chat_id]["msg_id"][-1]
                context.bot.editMessageText(text=send_text, chat_id=chat_id, message_id=last_msg, parse_mode="HTML")
                self.users[chat_id]["history"].pop()
                self.users[chat_id]["history"].pop()
                self.users[chat_id]["user_in"].pop()
                self.users[chat_id]["msg_id"].pop()
                # if there is previous message - add buttons to previous message
                if len(self.users[chat_id]["msg_id"]) > 0:
                    send_text = self.users[chat_id]["history"][-1]
                    message_id = self.users[chat_id]["msg_id"][-1]
                    context.bot.editMessageText(text=send_text, chat_id=chat_id,
                                                message_id=message_id, reply_markup=self.button)
                self.save_user_history(chat_id, self.users[chat_id]["name2"])
            else:
                send_text = msg_text + "\n<HISTORY LOST>"
                context.bot.editMessageText(text=send_text, chat_id=chat_id,
                                            message_id=msg_id, reply_markup=self.button)
        elif option == "Chars":
            char_list = self.get_characters_files_list()
            to_send = []
            for i, char in enumerate(char_list):
                to_send.append("/" + self.load_cmd + str(i) + " " + char.replace(".json", ""))
                if i % 50 == 0 and i != 0:
                    send_text = "\n".join(to_send)
                    context.bot.send_message(text=send_text, chat_id=chat_id)
                    to_send = []
            if len(to_send) > 0:
                send_text = "\n".join(to_send)
                context.bot.send_message(text=send_text, chat_id=chat_id)

    # =============================================================================
    # answer generator
    def generate_answer(self, user_in, chat_id):
        # if generation will fail, return "fail" answer
        answer = self.GENERATOR_FAIL
        # acquire generator lock if we can
        while self.generator_lock.locked():
            sleep(1)
        self.generator_lock.acquire(timeout=600)
        try:
            # Append user_in history
            self.users[chat_id]["user_in"].append(user_in)
            # Preprocessing: add user_in to history in right order:
            if self.bot_mode == "notebook":
                # If notebook mode - append to history only user_in, no additional preparing;
                self.users[chat_id]["history"].append(user_in)
            elif user_in.startswith(self.impersonate_prefix):
                # If user_in starts with prefix - impersonate-like (if you try to get "impersonate view")
                # adding "" line to prevent bug in history sequence, user_in is prefix for bot answer
                self.users[chat_id]["history"].append("")
                self.users[chat_id]["history"].append(user_in[len(self.impersonate_prefix):] + ":")
            elif user_in == "":
                # if user_in is "" - no user text, it is like continue generation
                # adding "" history line to prevent bug in history sequence, add "name2:" prefix for generation
                self.users[chat_id]["history"].append("")
                self.users[chat_id]["history"].append(self.users[chat_id]["name2"] + ":")
            else:
                # If not notebook/impersonate/continue mode then use ordinary chat preparing
                # add "name1&2:" to user and bot message (generation from name2 point of view);
                self.users[chat_id]["history"].append(self.users[chat_id]["name1"] + ":" + user_in)
                self.users[chat_id]["history"].append(self.users[chat_id]["name2"] + ":")
            # Set eos_token and stopping_strings.
            stopping_strings = []
            eos_token = None
            if self.bot_mode in ["chat", "chat-restricted"]:
                # don't know why, but better works without stopping_strings
                # stopping_strings = [f'\n{self.users[chat_id]["name2"]}:', f'\n{self.users[chat_id]["name1"]}:']
                eos_token = '\n'
            # Make prompt: context + conversation history
            prompt = self.users[chat_id]["context"] + "\n" + "\n".join(self.users[chat_id]["history"])
            # Generate!
            generator = generate_reply(
                question=prompt, state=self.generation_params,
                eos_token=eos_token, stopping_strings=stopping_strings
            )
            # This is "bad" implementation of getting answer
            for a in generator:
                answer = a
            # If generation result zero length - return  "Empty answer."
            if len(answer) < 1:
                answer = self.GENERATOR_EMPTY_ANSWER
        except Exception as e:
            print("generate_answer", e)
        finally:
            # anyway, release generator lock. Then return
            self.generator_lock.release()
            if answer not in [self.GENERATOR_EMPTY_ANSWER, self.GENERATOR_FAIL]:
                # if everything ok - add generated answer in history and return last message
                self.users[chat_id]["history"][-1] = self.users[chat_id]["history"][-1] + answer
                return self.users[chat_id]["history"][-1]
            else:
                return answer

    # =============================================================================
    # load characters char_file from ./characters
    def load_char_file(self, char_file:str):
        # Copy default user data. If reading will fail - return default user data
        user = deepcopy(self.default_users_data.copy())
        try:
            # Try to read char file.
            char_file_path = Path(f'{self.characters_dir_path}/{char_file}')
            with open(char_file_path, 'r', encoding='utf-8') as user_file:
                if char_file.split(".")[-1] == "json":
                    data = json.loads(user_file.read())
                else:
                    data = yaml.safe_load(user_file.read())
            #  load persona and scenario
            if 'you_name' in data: user["name1"] = data['you_name']
            if 'char_name' in data: user["name2"] = data['char_name']
            if 'name' in data: user["name2"] = data['name']
            if 'char_persona' in data:
                user["context"] += f"{data['char_name']}'s Persona: {data['char_persona'].strip()}\n"
            if 'world_scenario' in data:
                user["context"] += f"Scenario: {data['world_scenario'].strip()}\n"
            #  add dialogue examples
            if 'example_dialogue' in data:
                data['example_dialogue'] = data['example_dialogue'].replace('{{user}}', user["name1"])
                data['example_dialogue'] = data['example_dialogue'].replace('{{char}}', user["name2"])
                data['example_dialogue'] = data['example_dialogue'].replace('<USER>', user["name1"])
                data['example_dialogue'] = data['example_dialogue'].replace('<BOT>', user["name2"])
                user["context"] += f"{data['example_dialogue'].strip()}\n"
            #  add <START>, add char greeting
            user["context"] += f"{user['context'].strip()}\n<START>\n"
            if 'char_greeting' in data:
                user["context"] += '\n' + data['char_greeting'].strip()
                user["greeting"] = data['char_greeting'].strip()
            if 'greeting' in data:
                user["context"] += '\n' + data['greeting'].strip()
                user["greeting"] = data['greeting'].strip()
        except Exception as e:
            print("load_char_json_file", e)
        finally:
            return user


def run_server():
    # example with char load context:
    tg_server = TelegramBotWrapper(bot_mode=params['bot_mode'], default_char_json=params['character_to_load'])
    tg_server.run_telegram_bot()  # by default - read token from extensions/telegram_bot/telegram_token.txt


def setup():
    Thread(target=run_server, daemon=True).start()
