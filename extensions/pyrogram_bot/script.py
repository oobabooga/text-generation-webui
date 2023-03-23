import asyncio
from threading import Thread

import uvloop

from .client import Client


async def pyrogram_main() -> None:
  app = Client()
  await app.run()

def start_bot() -> None:
  uvloop.install()
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(pyrogram_main())

def ui() -> None:
  Thread(target=start_bot, daemon=True).start()

if __name__=="__main__":
  start_bot()
