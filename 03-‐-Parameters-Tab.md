## Generation

Contains parameters that control the text generation. 

### Quick rundown

LLMs work by generating one token at a time. Given your prompt, all the model does is calculate the probabilities for every possible next token. The actual token generation is done after that. 

* In *greedy decoding*, the most likely token is always picked.
* Most commonly, *sampling* techniques are used to choose from the next-token distribution in a more non-trivial way with the goal of improving the quality of the generated text.

### Preset menu

Can be used to save combinations of parameters for reuse. 

The built-in presets were not manually chosen. They were obtained after a blind contest where hundreds of people voted which I called "Preset Arena". The results can be found [here](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md).

A key takeaway is that the best presets are:

* **For Instruct**: Divine Intellect, Big O, simple-1, Space Alien, StarChat, Titanic, tfs-with-top-a, Asterism, Contrastive Search (only works for the Transformers loader at the moment).
* **For Chat**: Midnight Enigma, Yara, Shortwave.

The other presets are:

* Mirostat: a special decoding technique first implemented in llama.cpp and then adapted into this repository for all loaders. Many people have obtained positive results with it for chat.
* LLaMA-Precise: a legacy preset that was the default for the web UI before the Preset Arena.
* Debug-deterministic: disables sampling. It is useful for debugging, or if you intentionally want to use greedy decoding.

### Parameters description

For a technical description of the parameters, the [transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) is a good reference.

* **max_new_tokens**: Maximum number of tokens to generate. It is used in the truncation calculation through the formula `(prompt_length) = min(truncation_length - max_new_tokens, prompt_length)`, so don't set it higher than necessary.
* **temperature**: Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.
* **top_p**: If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.
* **top_k**: Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.
* **repetition_penalty**: Penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.
* **repetition_penalty_range**: The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used.
* **typical_p**: If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.
* **tfs**: Tries to detect a tail of low probability tokens in the distribution and disconsiders those tokens. See [this blog post](https://www.trentonbricken.com/Tail-Free-Sampling/) for details. The closer to 0, the more discarded tokens.
* **top_a**: Tokens with probability smaller than `(top_a) * (probability of the most likely token)^2` are discarded.
* **epsilon_cutoff**: In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled.
* **eta_cutoff**: In units of 1e-4; a reasonable value is 3. The main parameter of the special Eta Sampling technique. See [this paper](https://arxiv.org/pdf/2210.15191.pdf) for a description.
* **guidance_scale**: the main parameter for Classifier-Free Guidance (CFG). [The paper](https://arxiv.org/pdf/2306.17806.pdf) suggests that 1.5 is a good value. It can be used in conjunction with a negative prompt or not.
* **Negative prompt**: only used when `guidance_scale != 1`. It is most useful for instruct models and custom system messages. You place your full prompt with the default system message for the model (like "You are Llama, a helpful assistant...") in this field to make the model pay more attention to your custom message.
* **penalty_alpha**: Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.
* **mirostat_mode**: activates the Mirostat sampling technique. It aims to control perplexity during sampling. See the [paper](https://arxiv.org/abs/2007.14966).
* **mirostat_tau**: no idea, see the paper for details. According to the Preset Arena, 8 is a good value. 
* **mirostat_tau**: no idea, see the paper for details. According to the Preset Arena, 0.1 is a good value.
* **do_sample**: when unchecked, sampling is entirely disabled and greedy decoding is used instead (the most likely token is always picked).
* **Seed**: set the Pytorch seed to this number. Note that some loaders do not use Pytorch (notably llama.cpp), and others are not deterministic (notably ExLlama v1 and v2). For these loaders, the seed has no effect.
* **encoder_repetition_penalty**: Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.
* **no_repeat_ngram_size**: If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.
* **min_length**: Minimum generation length in tokens. This is a built-in parameter in the transformers library that has never been very useful. Typically you want to check "Ban the eos_token" instead.
* **num_beams**: Number of beams for beam search. 1 means no beam search.
* **length_penalty**: used by beam search only. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
* **early_stopping**: used by beam search only. When checked, where the generation stops as soon as there are "num_beams" complete candidates; otherwise, an heuristic is applied and the generation stops when is it very unlikely to find better candidates (I just copied this from the transformers documentation and have never gotten beam search to generate good results).

To the right (or below if you are on mobile), the following parameters are present:

* **Truncate the prompt up to this length**: used to prevent the prompt from getting bigger than the model's context length. In case of the transformers loader, which allocates memory dynamically, this parameter can also be used to set a VRAM ceiling and prevent out-of-memory errors. This parameter is automatically updated with the model's context length (from n_ctx or max_seq_len for loaders that use these parameters, and from the model metadata directly for loaders that don't) when you load a model.
* **Maximum number of tokens/second**: to make text readable in real time in case the model is generating too fast. Good if you want to flex and tell everyone how good your GPU is.
* **Custom stopping strings**: the model stops generating as soon as any of the strings set in this field is generated. Note that when generating text in the Chat tab, some default stopping strings are set regardless of this parameter, like "\nYour Name:" and "\nBot name:" for chat mode. That's why this parameter has a "Custom" in its name.
* **Custom token bans**: allows you to ban the model from generating certain tokens altogether. You need to find the token IDs under "Default" > "Tokens" or "Notebook" > "Tokens", or by looking at the `tokenizer.json` for the model directly.
* **auto_max_new_tokens**: when checked, the max_new_tokens parameter is expanded in the backend to the available context length. The maximum length is given by the "truncation_length" parameter. This is useful for getting long replies in the Chat tab without having to click on "Continue" many times.
* **Ban the eos_token**: one of the possible tokens that a model can generate is the EOS (End of Sequence) token. When it is generated, the generation stops prematurely. When this parameter is checked, that token is banned from being generated, and the generation will always generate "max_new_tokens" tokens.
* **Add the bos_token to the beginning of prompts**: by default, the model tokenizer will add a BOS (Beginning of Sequence) token to your prompt. During training, BOS tokens are used to separate different documents. If unchecked, no BOS token will be added, and the model will interpret your prompt as being in the middle of a document instead of at the start of one. This significantly changes the output and can make it more creative.
* **Skip special tokens**: when decoding the generated tokens, skip special tokens from being converted to their text representation. Otherwise, BOS appears as `<s>`, EOS as `</s>`, etc.
* **Activate text streaming**: when unchecked, the full response is outputted at once, without streaming the words one at a time. I recommend unchecking this parameter on high latency networks like running the webui on Google Colab or using `--share`.
* **Load grammar from file**: loads a GBNF grammar from a file under `text-generation-webui/grammars`. The output is written to the "Grammar" box below. You can also save and delete custom grammars using this menu.
* **Grammar**: allows you to constrain the model output to a particular format. For instance, you can make the model generate lists, JSON, specific words, etc. Grammar is extremely powerful and I use it all the time. The syntax looks a bit daunting at first sight, but it is very easy once you understand the syntax. See the [GBNF Guide](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) for details.
