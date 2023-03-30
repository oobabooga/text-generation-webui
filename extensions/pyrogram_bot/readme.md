# text-generation-webui telegram bot extension

Kudos to:

- [oobabooga](https://github.com/oobabooga) for awesome [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [delivrance](https://github.com/delivrance) for awesome [pyrogram](https://github.com/pyrogram/pyrogram)

To run this plugin make a few steps:

- Copy example environment variables to `.env` file

  ```bash
  cat extensions/pyrogram_bot/example.env >> .env
  ```

- Write your telegram bot credentials inside your `.env` file

- Come to plugin folder

  ```bash
  cd extensions/pyrogram_bot
  ```

- Be sure you in conda environment

  ```bash
  conda activate textgen
  ```

- Install dependencies

  ```bash
  pip install -r requirements.py
  ```
