This is a simple extension to connect text-generator to telegram in cai-chat-like style or notebook style.
-

REQUIREMENTS:
- python-telegram-bot==13.15

HOW TO USE:
1) place to your bot token to "text-generation-webui\extensions\telegram_bot\telegram_token.txt"
2) run server.py with "--extensions telegram_bot"

FEATURES:
- chat and notebook modes
- session for all users are separative (by chat_id)
- local session history - conversation won't be lost. Separated history between users and chars.
- nice "X typing" during generating (users will not think that bot is stuck)
- regenerate last message, remove last messages from history, reset history button, continue previous message
- you can load new characters from text-generation-webui\characters with "_LOAD:" command!!!
- chatting # prefix for impersonate: "#You" or "#Castle guard" or "#Alice thoughts about me"


TBC:
- replace "X typing" by yield from generator
- group chat mode (need to be tested, does current workflow is ok?)
- change characters during session - DONE! (due to "/load:" command)
- limit of simultaneously threading to prevent overloading - DONE! (one thread allowed, if someone find out that 2 generator thread faster - function will be rewrited) 
- history upgrade: local cache - DONE!
- message_id sequence history cache lead to "remove last message" button - DONE!
- separated telegram_settings.json file - DONE! (telegram_token.txt)
