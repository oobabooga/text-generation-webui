Used to have multi-turn conversations with the model.

## Input area

The following buttons can be found. Note that the hover menu can be replaced with always-visible buttons with the `--chat-buttons` flag.

* **Generate**: sends your message and makes the model start a reply.
* **Stop**: stops an ongoing generation as soon as the next token is generated (which can take a while for a slow model).
* **Continue**: makes the model attempt to continue the existing reply. In some cases, the model may simply end the existing turn immediately without generating anything new, but in other cases, it may generate a longer reply.
* **Regenerate**: similar to Generate, but your last message is used as input instead of the text in the input field. Note that if the temperature/top_p/top_k parameters are low in the "Parameters" tab of the UI, the new reply may end up identical to the previous one.
* **Remove last reply**: removes the last input/output pair from the history and sends your last message back into the input field.
* **Replace last reply**: replaces the last reply with whatever you typed into the input field. Useful in conjunction with "Copy last reply" if you want to edit the bot response.
* **Copy last reply**: sends the contents of the bot's last reply to the input field.
* **Impersonate**: makes the model generate a new message on your behalf in the input field, taking into consideration the existing chat history.
* **Send dummy message**: adds a new message to the chat history without causing the model to generate a reply.
* **Send dummy reply**: adds a new reply to the chat history as if the model had generated this reply. Useful in conjunction with "Send dummy message".
* **Start new chat**: starts a new conversation while keeping the old one saved. If you are talking to a character that has a "Greeting" message defined, this message will be automatically added to the new history.
* **Send to default**: sends the entire chat prompt up to now to the "Default" tab.
* **Send to notebook**: sends the entire chat prompt up to now to the "Notebook" tab.

The **Show controls** checkbox causes the input fields below the input textbox to disappear. It is useful for making the page fit entirely into view and not scroll.

## Past chats

Allows you to switch between the current and previous conversations with the current character, or between the current and previous instruct conversations (if in "instruct" mode). The **Rename** menu can be used to give a unique name to the selected conversation, and the 🗑️ button allows you to delete it.

## Start reply with

Whatever you type there will appear at the start of every reply by the bot. This is useful to guide the response in the desired direction.

## Mode

The most important input field. It defines how the chat prompt is formatted. There are 3 options: chat, chat-instruct, and instruct. It is worth going into more detail about this because it seems to not be obvious to a lot of people.

### Instruction-following models

There are two kinds of models: base models, like Llama and GPT-J, and fine-tuned models, like Alpaca and Vicuna. Fine-tuned models are trained starting from base models, most often with the goal of getting the model to understand and respond to instructions just like ChatGPT does. Let's call such models *instruction-following models*.

Each instruction-following model was trained on a specific prompt format, and you have to use that exact prompt format if you want the model to follow your instructions as accurately as it can.

As an example, this is the Alpaca format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Hi there!

### Response:
Hello! It's nice to meet you. What can I help with?

### Instruction:
How are you?

### Response:
I'm doing well, thank you for asking! Is there something specific you would like to talk about or ask me? I'm here to help answer any questions you may have.
```

This format is characterized by a context string at the top, and alternating turns where each user input starts with `### Instruction:` and each bot turn starts with `### Response:`. There are also weirder formats, like the one used by the Llama-2-chat models released by Meta AI:

```
[INST] <<SYS>>
Answer the questions.
<</SYS>>
Hi there! [/INST] Hello! It's nice to meet you. What can I help with? </s><s>[INST] How are you? [/INST] I'm doing well, thank you for asking! Is there something specific you would like to talk about or ask me? I'm here to help answer any questions you may have.
```

In this format, there are special tokens at the end of each bot reply (`</s>`, the end of sequence token, and `<s>`, the beginning of sequence token); no new lines separating the turns; and the context string is written between `<<SYS>>` and `<</SYS>>`. Despite the intimidating look of this format, the logic is the same: there are user turns and bot turns, and each one appears in a specific place in the template.

It is important to emphasize that instruction-following models **have to be used with the exact prompt format that they were trained on**. Using those models with any other prompt format should be considered undefined behavior. The model will still generate replies, but they will be less accurate to your inputs.

Now that an instruction-following model is defined, we can move on to describing the 3 chat modes.

### Chat

Used for talking to the character defined under "Parameters" > "Character" using a simple chat prompt in this format:

```
Chiharu Yamada's Persona: Chiharu Yamada is a young, computer engineer-nerd with a knack for problem solving and a passion for technology.
You: Hi there!
Chiharu Yamada: Hello! It's nice to meet you. What can I help with?
You: How are you?
Chiharu Yamada: I'm doing well, thank you for asking! Is there something specific you would like to talk about or ask me? I'm here to help answer any questions you may have.
```

There are 3 adjustable parameters in "Parameters" > "Character" being used in this prompt:

* The **Context** string appears at the top of the prompt. Most often it describes the bot's personality and adds a few example messages to guide the model towards the desired reply length and format. This string never gets truncated: as the prompt size increases, old messages get removed one at a time until the prompt becomes smaller than the truncation length set under "Parameters" > "Generation" > "Truncate the prompt up to this length".
* The **Your name** string appears at the beginning of each user reply. By default, this string is "You".
* The **Character's name** string appears at the beginning of each bot reply.

Additionally, the **Greeting** string appears as the bot's opening message whenever the history is cleared.

The "Chat" option should typically be used only for base models or non-instruct fine tunes, and should not be used for instruction-following models.

### Instruct

Used for talking to an instruction-following model using the prompt format defined under "Parameters" > "Instruction template". Think of this option as an offline ChatGPT.

The prompt format is defined by the **Instruction template** parameter in "Parameters" > "Instruction template", which represents a Jinja2 template.

Note that when you load a model in the "Model" tab, the web UI will try to automatically detect its instruction template (if any), and will update the values under "Parameters" > "Instruction template" accordingly. This is done using a set of regular expressions defined in `models/config.yaml`. This detection is not guaranteed to be accurate. You should check the model card on Hugging Face to see if you are using the correct prompt format.

### Chat-instruct

As said above, instruction-following models are meant to be used with their specific prompt templates. The chat-instruct mode allows you to use those templates to generate a chat reply, thus mixing Chat and Instruct modes (hence the name).

It works by creating a single instruction-following turn where a command is given followed by the regular chat prompt. Here is an example in Alpaca format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Continue the chat dialogue below. Write a single reply for the character "Chiharu Yamada".
Chiharu Yamada's Persona: Chiharu Yamada is a young, computer engineer-nerd with a knack for problem solving and a passion for technology.
You: Hi there!
Chiharu Yamada: Hello! It's nice to meet you. What can I help with?
You: How are you?

### Response:
Chiharu Yamada:
```

Here, the command is

> Continue the chat dialogue below. Write a single reply for the character "Chiharu Yamada".

Below this command, the regular chat prompt is added, including its Context string and the chat history, and then the user turn ends. The bot turn starts with the "Character's name" string followed by `:`, thus prompting the instruction-following model to write a single reply for the character.

The chat-instruct command can be customized under "Parameters" > "Instruction template" > "Command for chat-instruct mode". Inside that command string, `<|character|>` is a placeholder that gets replaced with the bot name, and `<|prompt|>` is a placeholder that gets replaced with the full chat prompt.

Note that you can get creative: instead of writing something trivial like "Write a single reply for the character", you could add more complex instructions like

> This is an adventure game, and your task is to write a reply in name of "<|character|>" where 3 options are given for the user to then choose from.

And it works:

![chat-instruct](https://github.com/oobabooga/text-generation-webui/assets/112222186/e38e3469-8263-4a10-b1a1-3c955026b8e7)

## Chat style

This defines the visual style of the chat UI. Each option is a CSS file defined under `text-generation-webui/css/chat_style-name.css`, where "name" is how this style is called in the dropdown menu. You can add new styles by simply copying `chat_style-cai-chat.css` to `chat_style-myNewStyle.css` and editing the contents of this new file. If you end up with a style that you like, you are highly encouraged to submit it to the repository.

The styles are only applied to chat and chat-instruct modes. Instruct mode has its separate style defined in `text-generation-webui/css/html_instruct_style.css`.

## Character gallery

This menu is a built-in extension defined under `text-generation-webui/extensions/gallery`. It displays a gallery with your characters, and if you click on a character, it will be automatically selected in the menu under "Parameters" > "Character".
