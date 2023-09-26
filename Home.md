🚧🚧 Under construction, come back later 🚧🚧

# Documentation

## What works

| Loader | Loading one LoRA | Training LoRAs | Multimodal extension | Perplexity evaluation |
|-----|-----|-----|-----|-----|
| Transformers | ✅ | ✅* | ✅ | ✅ | ✅ |
| ExLlama_HF | ✅ | ❌ | ❌ | ✅ | ✅ |
| ExLlamav2_HF | ❌ | ❌ | ❌ | ✅ | ✅ |
| ExLlama | ✅ | ❌ | ❌ | ❌ | ✅ |
| ExLlamav2 | ❌ | ❌ | ❌ | ❌ | ❌ |
| AutoGPTQ | ✅ | ❌ | ✅ | ✅ | ✅ |
| GPTQ-for-LLaMa | ✅** | ✅** | ✅ | ✅ | ✅ |
| llama.cpp | ❌ | ❌ | ❌ | ❌ | ❌ |
| llamacpp_HF | ❌ | ❌ | ❌ | ✅ | ✅ |
| ctransformers | ❌ | ❌ | ❌ | ❌ | ❌ |

✅ = implemented

❌ = not implemented

\* For training LoRAs with GPTQ models, use this loader with the options `auto_devices` and `disable_exllama` checked.

\*\* Needs the monkey-patch.

## Chat tab

Used to have multi-turn conversations with the model.

The following buttons can be found in this tab:

* **Generate**: sends your message and makes the model start a reply.
* **Stop**: stops an ongoing generation as soon as the next token is generated (which can take a while for a slow model).
* **Continue**: makes the model attempt to continue the existing reply. In some cases, the model may simply end the existing turn immediately without generating anything new, but in other cases it may generate a longer reply.
* **Impersonate**: makes the model generate a new message on your behalf in the input field, taking into consideration the existing chat history.
* **Regenerate**: similar to Generate, but your last message is used as input instead of the text in the input field. Note that if the temperature/top_p/top_k parameters are low in the "Parameters" tab of the UI, the new reply may end up identical to the previous one.
* **Remove last**: removes the last input/output pair from the history and sends your last message back into the input field.
* **Copy last reply**: sends the contents of the bot's last reply to the input field.
* **Replace last reply**: replaces the last bot reply with whatever you typed into the input field. Useful in conjunction with "Copy last reply" if you want to edit the bot response.
* **Send dummy message**: adds a new message to the chat history without causing the model to generate a reply.
* **Send dummy reply**: adds a new reply to the chat history as if the model had generated this reply. Useful in conjunction with "Send dummy message".
* **Clear chat history**: clears the history. After clicking on this button, it will split into two (Confirm and Cancel) to prevent you from accidentally clearing the history. If you are talking to a character that has a "Greeting" message defined, this message will be automatically added to the cleared history.
* **Send to default**: sends the entire chat prompt up to now to the "Default" tab.
* **Send to notebook**: sends the entire chat prompt up to now to the "Notebook" tab.

Below the buttons, you can find the following input fields:

### Start reply with

Whatever you type there will appear at the start of every reply by the bot. This is useful to guide the response in a desired direction.

### Mode

The most important input field. It defines how the chat prompt is formatted. There are 3 options: chat, chat-instruct, and instruct. It is worth going into more detail about this because it seems to not be obvious to a lot of people.

#### Instruction-following models

There are two kinds of models: base models, like Llama and GPT-J, and fine-tuned models, like Alpaca and Vicuna. Fine-tuned models are trained starting from base models, most often with the goal of getting the model to understand and respond to instructions just like ChatGPT does. Let's call such models *instruction-following models*.

Each instruction-following model was trained on a specific prompt format, and you have to use that exact same prompt format if you want the model to follow your instructions as accurately as it can.

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

In this format, there are special tokens at the end of each bot reply (`</s>`, the end of sequence token, and `<s>`, the beginning of sequence token), there are no new lines separating the turns, and the context string is written between `<<SYS>>` and `<</SYS>>`. Despite the intimidating look of this format, the logic is the same: there are user turns and bot turns, and each one is formatted in some speficic way.

It is important to emphasize that instruction-following models **have to be used with the exact prompt format that they were trained on**. Using those models with any other prompt format should be considered undefined behavior. The model will still generate replies, but they will be less accurate to your inputs since you will not be using the model like it was meant to.

Now that an instruction-following model is defined, we can move on to describing the 3 chat modes.

#### Chat

Used for talking to the character defined under "Parameters" > "Character" using a simple chat prompt in this format:

```
Chiharu Yamada's Persona: Chiharu Yamada is a young, computer engineer-nerd with a knack for problem solving and a passion for technology.

You: Hi there!
Chiharu Yamada: Hello! It's nice to meet you. What can I help with?
You: How are you?
Chiharu Yamada: I'm doing well, thank you for asking! Is there something specific you would like to talk about or ask me? I'm here to help answer any questions you may have.
```

There are 3 adjustable parameters in "Parameters" > "Character" being used in this prompt:

* The **Context** string appears at the top of the prompt. Most often it describes the bot's personality and adds a few example messages to guide the model towards the desired reply length and format. 
* The **Your name** string appears at the beginning of each user reply. By default, this string is "You".
* The **Character's name** string appears at the beginning of each bot reply.

Additionally, the **Greeting** string appears as the bot's opening message whenever the history is cleared.

The "Chat" option should typically be used only for base models, and should not be used for instruction-following models.

#### Instruct

Used for talking to an instruction-following model using the prompt format defined under "Parameters" > "Instruction template". Think of this option as an offline ChatGPT.

The prompt format is defined by the following adjustable parameters in "Parameters" > "Instruction template":

* **Context**: appears at the top of the prompt exactly as it is written, including the new line characters at the end if any. Often the context includes a customizable sub-string. For instance, instead of "Answer the questions." for Llama-2-chat, you can write "Answer the questions as if you were a pirate.", and the model will comply.
* **Turn template**: defines a single input/reply turn. In this string, `<|user|>` and `<|bot|>` are placeholders that get replaced with whatever you type in the **User string** and **Bot string** fields respectively, while `<|user-message|>` and `<|bot-message|>` get replaced with the user and bot messages at that turn. If the prompt format uses new line characters, they should be written inline as `\n` in the turn template.

Note that when you load a model in the "Model" tab, the web UI will try to automatically detect its instruction template (if any), and will update the values under "Parameters" > "Instruction template" accordingly. This is done using a set of regular expressions defined in `models/config.yaml`. This detection is not guaranteed to be accurate. You should check the model card on Hugging Face to see if you are using an instruction template that is correct and makes sense.

#### Chat-instruct

As said above, instruction-following models are meant to be used with their specific prompt templates. The chat-instruct mode allows you to use those templates to talk generate a chat reply, thus mixing Chat and Instruct modes (hence the name).

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

The chat-instruct command can be customized under "Parameters" > "Instruction template" > "Command for chat-instruct mode". Inside that command string, `<|character|>` is a placeholder that gets replaced with the bot name and `<|prompt|>` is a placeholder that gets replaced with the full chat prompt.

Note that you can get creative: instead of writing something trivial like "Write a single reply for the character", you could add more complex instructions like

> This is an adventure game, and your task is to write a reply in name of "<|character|>" where 3 options are given for the user to then choose from.

And it works:

[print]

### Chat style

This defines the visual style of the chat UI. Each option is a CSS file defined under `text-generation-webui/css/chat_style-name.css`, where "name" is how this style is called in the dropdown menu. You can add new styles by simply copying `chat_style-cai-chat.css` to `chat_style-myNewStyle.css` and editing the contents of this new file. If you do that and end up with a style that you like, you are highly encouraged to submit it in a Pull Request to the repository.

The styles are only applied to chat and chat-instruct modes. Instruct mode has its own separate style defined in `text-generation-webui/css/html_instruct_style.css`.

### Character gallery

This menu is actually a built-in extension defined under `text-generation-webui/extensions/gallery`. It displays a gallery with your characters, and if you click on a character, it will be automatically selected in the menu under "Parameters" > "Character".

## Default and notebook tabs

Used to generate raw completions starting from your prompt.

### Default tab

This tab contains two main text boxes: Input, where you enter your prompt, and Output, where the model output will appear.

#### Input

The number on the lower right of the Input box counts the number of tokens in the input. It gets updated whenever you update the input text as long as a model is loaded (otherwise there is no tokenizer to count the tokens).

Below the Input box, the following buttons can be found:

* **Generate**: starts a new generation.
* **Stop**: causes an ongoing generation to be stopped as soon as a the next token after that is generated.
* **Continue**: starts a new generation taking as input the text in the Output box.

In the **Prompt** menu, you can select from some predefined prompts defined under `text-generation-webui/prompts`. The 💾 button saves your current input as a new prompt, the 🗑️ button deletes the selected prompt, and the 🔄 button refreshes the list. If you come up with an interesting prompt for a certain task, you are welcome to submit it to the repository in a Pull Request.

#### Output

Four tabs can be found:

* **Raw**: where the raw text generated by the model appears.
* **Markdown**: it contains a "Render" button. You can click on it at any time to render the current output as markdown. This is particularly useful for models that generate LaTeX equations like GALACTICA.
* **HTML**: displays the output in an HTML style that is meant to be more pleasing to read. Its style is defined under `text-generation-webui/css/html_readable_style.css`, and improvements to this style are welcome.
* **Logits**: when you click on "Get next token probabilities", this tab displays the 25 most likely next tokens and their probabilities based on your current input. If "Use samplers" is checked, the probabilities will be the the ones after the parameters in the "Parameters" tab are applied. Otherwise, they will be the raw probabilities generated by the model.

### Notebook tab

Precisely the same thing as the Default tab, with the difference that the output appears in the same text box as the input.

## Parameters tab

### Generation

These parameters control the text generation. 

This tab contains two main text boxes: Input, where you enter your prompt, and Output, where the model output will appear.

#### Input

The number on the lower right of the Input box counts the number of tokens in the input. It gets updated whenever you update the input text as long as a model is loaded (otherwise there is no tokenizer to count the tokens).

Below the Input box, the following buttons can be found:

* **Generate**: starts a new generation.
* **Stop**: causes an ongoing generation to be stopped as soon as a the next token after that is generated.
* **Continue**: starts a new generation taking as input the text in the Output box.

In the **Prompt** menu, you can select from some predefined prompts defined under `text-generation-webui/prompts`. The 💾 button saves your current input as a new prompt, the 🗑️ button deletes the selected prompt, and the 🔄 button refreshes the list. If you come up with an interesting prompt for a certain task, you are welcome to submit it to the repository in a Pull Request.

#### Output

Four tabs can be found:

* **Raw**: where the raw text generated by the model appears.
* **Markdown**: it contains a "Render" button. You can click on it at any time to render the current output as markdown. This is particularly useful for models that generate LaTeX equations like GALACTICA.
* **HTML**: displays the output in an HTML style that is meant to be more pleasing to read. Its style is defined under `text-generation-webui/css/html_readable_style.css`, and improvements to this style are welcome.
* **Logits**: when you click on "Get next token probabilities", this tab displays the 25 most likely next tokens based on your current input and their probabilities. If "Use samplers" is checked, the probabilities will be the the ones after the parameters in the "Parameters" tab are applied. Otherwise, they will be the raw probabilities generated by the model.

### Notebook tab

Precisely the same thing as the Default tab, with the difference that the output appears in the same text box as the input.

## Using LoRAs with GPTQ-for-LLaMa

This requires using a monkey patch that is supported by this web UI: https://github.com/johnsmith0031/alpaca_lora_4bit

To use it:

1. Install alpaca_lora_4bit using pip

```
git clone https://github.com/johnsmith0031/alpaca_lora_4bit.git
cd alpaca_lora_4bit
git fetch origin winglian-setup_pip
git checkout winglian-setup_pip
pip install .
```

2. Start the UI with the `--monkey-patch` flag:

```
python server.py --model llama-7b-4bit-128g --listen --lora tloen_alpaca-lora-7b --monkey-patch
```