import asyncio, threading
from os import getcwd, environ as env
from os.path import dirname, relpath

from modules import shared

from .services.i18n import get_i18n

path = dirname(relpath(__file__, getcwd()))
lang = "en"
t = get_i18n(lang)

try:
  from pyrogram import Client, filters, types
  from pyrogram.errors.exceptions import bad_request_400
  from dotenv import load_dotenv
  load_dotenv()

except ModuleNotFoundError:
  raise ModuleNotFoundError(t("error.dependencies_not_installed", { "/path/": path }))

class Client(Client):
  filters = filters
  types = types
  cwd = path
  lang = lang
  t = t
  owner = int(env["TELEGRAM_BOT_OWNER_ID"])
  allowed_chat = int(env["TELEGRAM_CHAT_ID"])

  def __init__(self, **kwargs) -> Client:
    try:
      shared.character = "Example"

      creds = {
        **kwargs,
        "api_id": env["TELEGRAM_API_ID"],
        "api_hash": env["TELEGRAM_API_HASH"],
        "bot_token": env["TELEGRAM_BOT_TOKEN"],
        "plugins": {
          "root": f"{path}/controllers"
        }
      }

      super().__init__(f"{path}/textgen", **creds)
    except bad_request_400.ApiIdInvalid:
      raise Exception(t("error.credential_file.is_invalid"))

  def change_lang(self, lang="en") -> None:
    self.lang = lang
    self.t = get_i18n(lang)

  async def run(self, stop_event: threading.Event):
    async with self:
      await self.start()

      while not stop_event.is_set():
        await asyncio.sleep(1)
      else:
        print("stopping", flush=True)
        await self.stop()
