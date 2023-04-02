This is extension to connect text-generator to telegram in cai-chat or notebook style.
-

REQUIREMENTS:
- python-telegram-bot==13.15

HOW TO USE:
1) open text-generation-webui\extensions\telegram_bot\script.py
2) replace PLACE_TELEGRAM_TOKEN_HERE to your bot token
3) you may set character_to_load (load from text-generation-webui\characters) and other params
4) run server.py with "--extensions telegram_bot"

FEATURES:
- chat and notebook modes
- session (chat history) for all users are separative (by chat_id)
- nice "X is typing" during generating (users will not think that bot is stuck)
- you can use .json characters!!!
- reset history button, remove single bot (not yours) message from history, continue bot's previous message
- threading: you can send few message simultaneously and bot will answer them all - but GPU may be overloaded and some message may stuck!!!

TBC:
- group chat mode (now all chat members is one entity "you" at bot perception)
- change characters during session
- find out something about limit of simultaneously threading to not overload GPU. Queue or something other.