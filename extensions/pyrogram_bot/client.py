import asyncio
from os import getcwd
from os.path import dirname, relpath

from modules import shared

from .services.i18n import get_i18n
from .services.get_creds import get_creds

path = dirname(relpath(__file__, getcwd()))
t = get_i18n()

try:
  from pyrogram import Client, filters, types
  from pyrogram.errors.exceptions import bad_request_400
except ModuleNotFoundError:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', { '/path/': path }))

class Client(Client):
  filters = filters
  types = types
  cwd = path

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
      return super()(f"{path}/textgen", **creds)
    except bad_request_400.ApiIdInvalid:
      raise Exception(t('error.credential_file.is_invalid'))

  def start(self) -> None:
    self.t = t
    super().start()

  def change_lang(self) -> None:
    self.t = get_i18n()

  async def run(self):
    async with self:
      try:
        await self.start()
      except ConnectionError:
        pass
      while True:
        await asyncio.sleep(1)
