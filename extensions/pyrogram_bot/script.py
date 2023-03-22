import asyncio
from os import getcwd
from os.path import dirname, relpath
from threading import Thread

from modules import shared

from .src.i18n import get_i18n
from .src.get_creds import get_creds

path = dirname(relpath(__file__, getcwd()))

t = get_i18n()

try:
  import uvloop
  from pyrogram import Client
  from pyrogram.errors.exceptions import bad_request_400
except ModuleNotFoundError:
  raise ModuleNotFoundError(t('error.dependencies_not_installed', {'/path/': path}))

async def pyrogram_main() -> None:
  try:
    creds = {
      **get_creds(path),
      "plugins": {"root": f"{path}/plugins"}
    }
    app = Client(f"{path}/textgen", **creds)
    shared.character = 'Example'
  except bad_request_400.ApiIdInvalid:
    raise Exception(t('error.credential_file.is_invalid'))

  async with app:
    try:
      await app.start()
    except ConnectionError:
      pass
    while True:
      await asyncio.sleep(1)

def start_bot() -> None:
  uvloop.install()
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(pyrogram_main())

def ui() -> None:
  Thread(target=start_bot, daemon=True).start()

if __name__=="__main__":
  start_bot()
