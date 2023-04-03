Custom chat mode characters are defined by `.yaml` files inside the `characters` folder.

The following fields may be defined:

| Field | Description |
|-------|-------------|
| `name` | The character's name. |
| `context` | A string that appears at the top of the prompt. It usually contains a description of the character's personality. |
| `greeting` (optional) | The character's opening message when a new conversation is started. |
| `example_dialogue` (optional) | A few example messages to guide the model. |

An example is included: [Example.yaml](https://github.com/oobabooga/text-generation-webui/blob/main/characters/Example.yaml)

#### How do I add a profile picture for my character?

You have two options:

* Put an image with the same name as your character's yaml file into the `characters` folder. For example, if your bot is `Character.yaml`, add `Character.jpg` or `Character.png` to the folder.
* Put an image called `img_bot.jpg` or `img_bot.png` into the `text-generation-webui` folder. This image will be used as the profile picture for any bots that don't have one.

#### How do I add a profile picture for myself?

Put an image called `img_me.jpg` or `img_me.png` into the `text-generation-webui` folder.

#### Is the chat history truncated in the prompt?

Once your prompt reaches the 2048 token limit, old messages will be removed one at a time. The context string will always stay at the top of the prompt and will never get truncated.

#### Pygmalion format characters

These are also supported out of the box. Simply put the JSON file in the `characters` folder, or upload it directly from the web UI by clicking on the "Upload character" tab at the bottom.