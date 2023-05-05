Custom chat mode characters are defined by `.yaml` files inside the `characters` folder. An example is included: [Example.yaml](https://github.com/oobabooga/text-generation-webui/blob/main/characters/Example.yaml)

The following fields may be defined:

| Field | Description |
|-------|-------------|
| `name` or `bot` | The character's name. |
| `your_name` or `user` (optional) | Your name. This overwrites what you had previously written in the `Your name` field in the interface. |
| `context` | A string that appears at the top of the prompt. It usually contains a description of the character's personality. |
| `greeting` (optional) | The character's opening message when a new conversation is started. |
| `example_dialogue` (optional) | A few example messages to guide the model. |
| `turn_template` (optional) | Used to define where the spaces and new line characters should be in Instruct mode. See the characters in `characters/instruction-following` for examples. |

#### Special tokens

* `{{char}}` or `<BOT>`: are replaced with the character's name
* `{{user}}` or `<USER>`: are replaced with your name

These replacements happen when the character is loaded, and they apply to the `context`, `greeting`, and `example_dialogue` fields.

#### How do I add a profile picture for my character?

Put an image with the same name as your character's yaml file into the `characters` folder. For example, if your bot is `Character.yaml`, add `Character.jpg` or `Character.png` to the folder.

#### Is the chat history truncated in the prompt?

Once your prompt reaches the 2048 token limit, old messages will be removed one at a time. The context string will always stay at the top of the prompt and will never get truncated.

#### Pygmalion format characters

These are also supported out of the box. Simply put the JSON file in the `characters` folder, or upload it directly from the web UI by clicking on the "Upload character" tab at the bottom.