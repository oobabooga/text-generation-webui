## Chat characters

Custom chat mode characters are defined by `.yaml` files inside the `characters` folder. An example is included: [Example.yaml](https://github.com/oobabooga/text-generation-webui/blob/main/characters/Example.yaml).

The following fields may be defined:

| Field | Description |
|-------|-------------|
| `name` or `bot` | The character's name. |
| `context` | A string that appears at the top of the prompt. It usually contains a description of the character's personality and a few example messages. |
| `greeting` (optional) | The character's opening message. It appears when the character is first loaded or when the history is cleared. |
| `your_name` or `user` (optional) | Your name. This overwrites what you had previously written in the `Your name` field in the interface. |

#### Special tokens

The following replacements happen when the prompt is generated, and they apply to the `context` and `greeting` fields:

* `{{char}}` and `<BOT>` get replaced with the character's name.
* `{{user}}` and `<USER>` get replaced with your name.

#### How do I add a profile picture for my character?

Put an image with the same name as your character's `.yaml` file into the `characters` folder. For example, if your bot is `Character.yaml`, add `Character.jpg` or `Character.png` to the folder.

#### Is the chat history truncated in the prompt?

Once your prompt reaches the `truncation_length` parameter (2048 by default), old messages will be removed one at a time. The context string will always stay at the top of the prompt and will never get truncated.

## Chat styles

Custom chat styles can be defined in the `text-generation-webui/css` folder. Simply create a new file with name starting in `chat_style-` and ending in `.css` and it will automatically appear in the "Chat style" dropdown menu in the interface. Examples:

```
chat_style-cai-chat.css
chat_style-TheEncrypted777.css
chat_style-wpp.css
```

You should use the same class names as in `chat_style-cai-chat.css` in your custom style.