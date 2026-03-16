## Generation

Contains parameters that control the text generation. 

### Quick rundown

LLMs work by generating one token at a time. Given your prompt, the model calculates the probabilities for every possible next token. The actual token generation is done after that. 

* In *greedy decoding*, the most likely token is always picked.
* Most commonly, *sampling* techniques are used to choose from the next-token distribution in a more non-trivial way with the goal of improving the quality of the generated text.

### Preset menu

Can be used to save and load combinations of parameters for reuse.

* **🎲 button**: creates a random yet interpretable preset. Only 1 parameter of each category is included for the categories: removing tail tokens, avoiding repetition, and flattening the distribution. That is, top_p and top_k are not mixed, and neither are repetition_penalty and frequency_penalty. You can use this button to break out of a loop of bad generations after multiple "Regenerate" attempts.

#### Built-in presets

These were obtained after a blind contest called "Preset Arena" where hundreds of people voted. The full results can be found [here](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md).

A key takeaway is that the best presets are:

* **For Instruct**: Divine Intellect, Big O, simple-1.
* **For Chat**: Midnight Enigma, Yara, Shortwave.

The other presets are:

* Mirostat: a special decoding technique first implemented in llama.cpp and then adapted into this repository for all loaders. Many people have obtained positive results with it for chat.
* LLaMA-Precise: a legacy preset that was the default for the web UI before the Preset Arena.
* Debug-deterministic: disables sampling. It is useful for debugging, or if you intentionally want to use greedy decoding.

### Parameters description

For more information about the parameters, the [transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) is a good reference.

* **max_new_tokens**: Maximum number of tokens to generate. Don't set it higher than necessary: it is used in the truncation calculation through the formula `(prompt_length) = min(truncation_length - max_new_tokens, prompt_length)`, so your prompt will get truncated if you set it too high.
* **temperature**: Primary factor to control the randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.
* **top_p**: If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.
* **min_p**: Tokens with probability smaller than `(min_p) * (probability of the most likely token)` are discarded. This is the same as top_a but without squaring the probability.
* **top_k**: Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.
* **repetition_penalty**: Penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.
* **presence_penalty**: Similar to repetition_penalty, but with an additive offset on the raw token scores instead of a multiplicative factor. It may generate better results. 0 means no penalty, higher value = less repetition, lower value = more repetition. Previously called "additive_repetition_penalty".
* **frequency_penalty**: Repetition penalty that scales based on how many times the token has appeared in the context. Be careful with this; there's no limit to how much a token can be penalized.
* **repetition_penalty_range**: The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used.
* **dry_multiplier**: Set to greater than 0 to enable DRY (Don't Repeat Yourself) sampling. It penalizes tokens that would extend a sequence that already appeared in the context. Recommended value: 0.8.
* **dry_allowed_length**: The longest sequence that can be repeated without being penalized by DRY. Shorter values make DRY more aggressive.
* **dry_base**: Controls how fast the DRY penalty grows with increasing sequence length.
* **typical_p**: If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.
* **tfs**: Tries to detect a tail of low-probability tokens in the distribution and removes those tokens. See [this blog post](https://www.trentonbricken.com/Tail-Free-Sampling/) for details. The closer to 0, the more discarded tokens.
* **top_a**: Tokens with probability smaller than `(top_a) * (probability of the most likely token)^2` are discarded.
* **top_n_sigma**: Keeps only tokens within N standard deviations of the mean log-probability. Acts as an adaptive cutoff that adjusts to the shape of the distribution. 0 disables it.
* **xtc_threshold**: eXclusion from Top Choices (XTC) sampling. If 2 or more tokens have probability above this threshold, the top token may be removed. This encourages the model to use less common word choices and can increase creativity.
* **xtc_probability**: The probability that XTC removal will actually happen when the threshold condition is met. Set to 1 for it to always apply, or lower for occasional application.
* **epsilon_cutoff**: In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled.
* **eta_cutoff**: In units of 1e-4; a reasonable value is 3. The main parameter of the special Eta Sampling technique. See [this paper](https://arxiv.org/pdf/2210.15191.pdf) for a description.
* **guidance_scale**: The main parameter for Classifier-Free Guidance (CFG). [The paper](https://arxiv.org/pdf/2306.17806.pdf) suggests that 1.5 is a good value. It can be used in conjunction with a negative prompt or not.
* **Negative prompt**: Only used when `guidance_scale != 1`. It is most useful for instruct models and custom system messages. You place your full prompt in this field with the system message replaced with the default one for the model (like "You are Llama, a helpful assistant...") to make the model pay more attention to your custom system message.
* **penalty_alpha**: Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.
* **mirostat_mode**: Activates Mirostat sampling, an adaptive decoding method that dynamically controls output perplexity for higher-quality text generation. 0 is disabled. 1 is the classic Mirostat algorithm described in [the paper](https://arxiv.org/abs/2007.14966), but can be less stable, or “wobbly,” and produce less coherent text. 2 is the improved version that is more stable and has lower perplexity, recommended for most use cases. 
*Note: Use either mirostat or dynamic_temperature, not both at the same time.*
* **mirostat_tau**: Target perplexity for Mirostat sampling. Controls how “surprising” the text is. Higher values = more diverse, lower = more predictable. Preset Arena suggests 8 as a good value.
* **mirostat_eta**: Learning rate for Mirostat’s perplexity adjustment. Higher values = adapts faster but less stable, lower values = slower but more stable. Preset Arena suggests 0.1 as a good value.
* **adaptive_target**: Target probability for adaptive-p sampling. This method adjusts the sampling threshold dynamically based on an exponential moving average of recent token probabilities. 0 disables it.
* **adaptive_decay**: EMA decay rate for adaptive-p sampling. Controls how quickly the running average adjusts. Default: 0.9.
* **dynamic_temperature**: Activates Dynamic Temperature. This modifies temperature to range between "dynatemp_low" (minimum) and "dynatemp_high" (maximum), with an entropy-based scaling. The steepness of the curve is controlled by "dynatemp_exponent". 
*Note: Use either dynamic_temperature or mirostat, not both at the same time.*
* **smoothing_factor**: Activates Quadratic Sampling. When `0 < smoothing_factor < 1`, the logits distribution becomes flatter. When `smoothing_factor > 1`, it becomes more peaked.
* **smoothing_curve**: Adjusts the dropoff curve of Quadratic Sampling. Higher values make the curve steeper. Only takes effect when smoothing_factor is set.
* **temperature_last**: Makes temperature the last sampler instead of the first. With this, you can remove low probability tokens with a sampler like min_p and then use a high temperature to make the model creative without losing coherency. Note: this parameter takes precedence over "Sampler priority". That means that `temperature`/`dynamic_temperature`/`quadratic_sampling` will be removed from wherever they are and moved to the end of the stack.
* **do_sample**: When unchecked, sampling is entirely disabled, and greedy decoding is used instead (the most likely token is always picked).
* **Seed**: Set the Pytorch seed to this number. Note that some loaders do not use Pytorch (notably llama.cpp). For these loaders, the seed has no effect.
* **encoder_repetition_penalty**: Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.
* **no_repeat_ngram_size**: If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.

To the right (or below if you are on mobile), the following parameters are present:

* **Truncate the prompt up to this length**: Used to prevent the prompt from getting bigger than the model's context length. In the case of the transformers loader, which allocates memory dynamically, this parameter can also be used to set a VRAM ceiling and prevent out-of-memory errors. This parameter is automatically updated with the model's context length (from "ctx_size" for loaders that use this parameter, and from the model metadata directly for loaders that do not) when you load a model.
* **Maximum number of tokens/second**: to make text readable in real-time in case the model is generating too fast. Good if you want to flex and tell everyone how good your GPU is.
* **Custom system message**: If not empty, will be used instead of the default system message in the instruction template. Useful for customizing the personality of the chatbot. Example: "You are a duck."
* **Custom stopping strings**: The model stops generating as soon as any of the strings set in this field is generated. Note that when generating text in the Chat tab, some default stopping strings are set regardless of this parameter, like "\nYour Name:" and "\nBot name:" for chat mode. That's why this parameter has a "Custom" in its name.
* **Custom token bans**: Allows you to ban the model from generating certain tokens altogether. You need to find the token IDs under "Default" > "Tokens" or "Notebook" > "Tokens", or by looking at the `tokenizer.json` for the model directly.
* **auto_max_new_tokens**: When checked, the max_new_tokens parameter is expanded in the backend to the available context length. The maximum length is given by the "truncation_length" parameter. This is useful for getting long replies in the Chat tab without having to click on "Continue" many times.
* **Ban the eos_token**: One of the possible tokens that a model can generate is the EOS (End of Sequence) token. When it is generated, the generation stops prematurely. When this parameter is checked, that token is banned from being generated, and the generation will always generate "max_new_tokens" tokens.
* **Add the bos_token to the beginning of prompts**: By default, the tokenizer will add a BOS (Beginning of Sequence) token to your prompt. During training, BOS tokens are used to separate different documents. If unchecked, no BOS token will be added, and the model will interpret your prompt as being in the middle of a document instead of at the start of one. This significantly changes the output and can make it more creative.
* **Skip special tokens**: When decoding the generated tokens, skip special tokens from being converted to their text representation. Otherwise, BOS appears as `<s>`, EOS as `</s>`, etc.
* **prompt_lookup_num_tokens**: Activates Prompt Lookup Decoding, a form of speculative decoding for the Transformers loader. It guesses future tokens by looking for matching patterns in the prompt itself, which can speed up generation for tasks that involve repeating or paraphrasing parts of the input.
* **Activate text streaming**: When unchecked, the full response is outputted at once, without streaming the words one at a time. I recommend unchecking this parameter on high latency networks like running the webui on Google Colab or using `--share`.
* **Static KV cache**: Use a static cache for improved performance with the Transformers loader. May not be compatible with all models.
* **Sampler priority**: Allows you to customize the order in which the different samplers are applied. The first sampler on the list gets applied first. With this, custom orders like `top_p -> temperature -> top_k` can be defined.
* **DRY sequence breakers**: Tokens across which DRY sequence matching is not continued. Typically punctuation and special tokens. Only used when DRY is active (dry_multiplier > 0).
* **Load grammar from file**: Loads a GBNF grammar from a file under `user_data/grammars`. The output is written to the "Grammar" box below. You can also save and delete custom grammars using this menu.
* **Grammar**: Allows you to constrain the model output to a particular format. For instance, you can make the model generate lists, JSON, specific words, etc. Grammar is extremely powerful and I highly recommend it. The syntax looks a bit daunting at first sight, but it gets very easy once you understand it. See the [GBNF Guide](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) for details.

### Chat tab controls

The following parameters appear in the Chat tab sidebar rather than the Parameters tab:

* **reasoning_effort**: Controls the thinking depth for models that support it (used by GPT-OSS). Options: low, medium, high.
* **enable_thinking**: Enables extended thinking mode for models that support it (used by Seed-OSS and pre-2507 Qwen3). When enabled, the model can use a thinking step before generating its reply.

## Instruction template

This sub-tab within the Parameters tab defines the instruction template used in the Chat tab when "instruct" or "chat-instruct" are selected under "Mode".

* **Saved instruction templates**: A dropdown menu where you can select a template. Click **Load** to apply it. The 💾 button saves the current template, and the 🗑️ button deletes the selected one.
* **Instruction template**: A Jinja2 template that defines the prompt format for the instruction-following conversation.
* **Send to notebook**: Send the full instruction template in string format to the Notebook tab.
* **Chat template**: A Jinja2 template that defines the prompt format for regular chat conversations with characters.

## Character tab

The Character tab is a separate top-level tab that contains the following sub-tabs:

### Character

Parameters that define the character used in the Chat tab when "chat" or "chat-instruct" are selected under "Mode".

* **Character**: A dropdown menu where you can select from saved characters, save a new character (💾 button), and delete the selected character (🗑️). The **Restore character** button resets the character to its last saved state.
* **Character's name**: The bot name as it appears in the prompt.
* **Context**: A string that is always at the top of the prompt. It never gets truncated. It usually defines the bot's personality and some key elements of the conversation.
* **Greeting**: An opening message for the bot. When set, it appears whenever you start a new chat.
* **Character picture**: A profile picture for the bot. To make it apply, you need to save the bot by clicking on 💾.
* **Your picture**: Your profile picture. It will be used in all conversations.

Note: the following replacements take place in the context and greeting fields when the chat prompt is generated:

* `{{char}}` and `<BOT>` get replaced with "Character's name".
* `{{user}}` and `<USER>` get replaced with "Your name".

So you can use those special placeholders in your character definitions. They are commonly found in TavernAI character cards.

### User

Allows you to create and manage user profiles.

* **User**: A dropdown to select, save (💾), or delete (🗑️) user profiles.
* **Name**: Your name as it appears in the prompt.
* **Description**: An optional description of yourself that can be referenced in conversations.

### Chat history

In this tab, you can download the current chat history in JSON format and upload a previously saved chat history.

When a history is uploaded, a new chat is created to hold it. That is, you don't lose your current chat in the Chat tab.

### Upload character

#### YAML or JSON

Allows you to upload characters in the YAML format used by the web UI, including optionally a profile picture.

#### TavernAI PNG

Allows you to upload a TavernAI character card. It will be converted to the internal YAML format of the web UI after upload.
