#### How to import my character?

Place the JSON file into the `characters` folder, or upload directly from the web UI by clicking on the "Upload character" tab at the bottom of the interface.

#### How to add a profile picture for my character?

You have two options:

* Place an image with the same name as your character's JSON file into the `characters` folder. For instance, `Character.jpg` if your bot is `Character.json`.
* Place an image called `img_bot.jpg` into the `text-generation-webui` folder. This image will be used for ALL bots that do not have their own profile picture. 

#### How to add a profile picture for myself?

Put an image called `img_you.jpg` into the `text-generation-webui` folder.

#### I have imported my character and can't see the `example_dialogue`.

The `example_dialogue` from your JSON character is parsed, added to the chat history, and kept hidden. You can check it by downloading your chat history and seeing the lines before `<|BEGIN-VISIBLE-CHAT|>`, or by starting the web UI with the `--verbose` option that will make your prompts be printed in the terminal.

Note that for the example dialogue to be parsed correctly, your name in the web UI (usually `You`) must match your name in the example dialogue.

#### Where are the `world_scenario` and `char_persona` fields?

Those are simply added to the the Context field when you load a character.

#### How can I create a character?

[Pygmalion JSON character creator](https://oobabooga.github.io/character-creator.html)

#### Is the chat history truncated in the prompt?

Old messages are removed one at a time once your prompt reaches the 2048 tokens limit. The context string is never truncated out and is always kept at the top of the prompt. 

#### I am running pygmalion-6b locally and my responses are really short. Why is that?

Try using the first commit of the model, which can be downloaded like this:

`python download-model.py PygmalionAI/pygmalion-6b --branch b8344bb4eb76a437797ad3b19420a13922aaabe1`

See these discussions for more information: [GitHub](https://github.com/oobabooga/text-generation-webui/issues/14), [HuggingFace](https://huggingface.co/PygmalionAI/pygmalion-6b/discussions/8#63d09347623a3d1d1174efa9)