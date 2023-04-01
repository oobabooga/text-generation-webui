import asyncio
from threading import Thread, Event

import dotenv, uvloop
dotenv.load_dotenv()

from modules import shared
shared.character = "Example"

from .client import Client

running = Event()

def establish_loop() -> None:
  running.set()

  uvloop.install()
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(Client().run())

def setup() -> None:
  if not running.is_set():
    Thread(target=establish_loop, daemon=True).start()
