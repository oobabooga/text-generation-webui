import asyncio, sqlite3, threading
from os import getcwd, remove
from os.path import dirname, relpath

from modules import shared

from .services.i18n import get_i18n
from .services.get_creds import get_creds

path = dirname(relpath(__file__, getcwd()))
lang = 'en'
t = get_i18n(lang)

try:
  from pyrogram import Client, filters, types
  from pyrogram.errors.exceptions import bad_request_400
except ModuleNotFoundError:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', { '/path/': path }))

class Client(Client):
  filters = filters
  types = types
  cwd = path
  lang = lang
  t = t

  def __init__(self, **kwargs) -> Client:
    try:
      shared.character = 'Example'
      creds = {
        **kwargs,
        **get_creds(path),
        "plugins": {
          "root": f"{path}/controllers"
        }
      }
      return super().__init__(f"{path}/textgen", **creds)
    except bad_request_400.ApiIdInvalid:
      raise Exception(t('error.credential_file.is_invalid'))

  def change_lang(self, lang='en') -> None:
    self.lang = lang
    self.t = get_i18n(lang)

  async def run(self, stop_event: threading.Event):
    async with self:
      await self.start()

      while not stop_event.is_set():
        await asyncio.sleep(1)
      else:
        print('stopping', flush=True)
        await self.stop()
