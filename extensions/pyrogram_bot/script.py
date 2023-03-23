import asyncio, atexit, threading, uvloop

from .client import Client

stop_event = threading.Event()
running = threading.Event()

async def pyrogram_main() -> None:
  running.set()
  app = Client()
  await app.run(stop_event)

def start_bot() -> None:
  uvloop.install()
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(pyrogram_main())

def exit_handle():
  print('\nGracefully stopping pyrogram bot.\n', flush=True)
  stop_event.set()

def ui() -> None:
  if not running.is_set():
    threading\
      .Thread(target=start_bot, daemon=True, name="pyrogram_thread")\
      .start()
    atexit.register(exit_handle)

if __name__=="__main__":
  start_bot()
