This is a simple extension to connect text-generator to telegram in cai-chat-like style or notebook style.
-

REQUIREMENTS:
- python-telegram-bot==13.15

HOW TO USE:
1) open text-generation-webui\extensions\telegram_bot\script.py
2) replace TELEGRAM_TOKEN to your bot token
3) you may set character_to_load (load from text-generation-webui\characters) and other params
4) run server.py with "--extensions telegram_bot"

FEATURES:
- chat and notebook modes
- session (chat history) for all users are separative (by chat_id)
- nice "X typing" during generating (users will not think that bot is stuck)
- you can use .json characters!!!
- reset history button, remove single bot (not yours) message from history, continue bot's previous message
- threading: you can send few message simultaneously and bot will answer them all - but GPU may be overloaded and some message may stuck!!!

TBC:
- replace "X typing" by yield from generator
- group chat mode (now all chat members is one entity "you" at bot perception)
- change characters during session
- limit of simultaneously threading to prevent overloading. Queue or something other. 
- history uprade: loacal cache 
- message_id sequence history cache lead to "remove last message" button 
