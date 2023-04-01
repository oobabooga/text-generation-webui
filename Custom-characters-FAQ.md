JSON characters are currently based on the format used by the Pygmalion model.

#### How to import my character?

Put the JSON file in the `characters` folder, or upload it directly from the web UI by clicking on the "Upload character" tab at the bottom.

#### How do I add a profile picture for my character?

You have three options:

* Upload any image (any format, any size) along with your JSON directly in the web UI.
* Put an image with the same name as your character's JSON file into the `characters` folder. For example, if your bot is `Character.json`, add `Character.jpg` or `Character.png` to the folder.
* Put an image called `img_bot.jpg` or `img_bot.png` into the `text-generation-webui` folder. This image will be used as the profile picture for any bots that don't have one.

#### How do I add a profile picture for myself?

Put an image called `img_me.jpg` or `img_me.png` into the `text-generation-webui` folder.

#### Where are the `world_scenario` and `char_persona` fields?

Those are simply added to the the Context field when you load a character.

#### How can I create a character?

* [JSON character creator](https://oobabooga.github.io/character-creator.html)
* [AI Character Editor](https://zoltanai.github.io/character-editor/)

#### Is the chat history truncated in the prompt?

Once your prompt reaches the 2048 token limit, old messages will be removed one at a time. The context string will always stay at the top of the prompt and will never be truncated.
