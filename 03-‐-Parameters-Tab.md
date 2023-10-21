## Parameters tab

### Generation

Contains parameters that control the text generation. 

#### Quick rundown

LLMs work by generating one token at a time. Given your prompt, all the model does is calculate the probabilities for every possible next token. The actual token generation is done after that. 

* In *greedy decoding*, the most likely token is always picked.
* Most commonly, *sampling* techniques are used to choose from the next-token distribution in a more non-trivial way with the goal of improving the quality of the generated text.

#### Preset menu

Can be used to save combinations of parameters for reuse. 

The built-in presets were not manually chosen. They were obtained after a blind contest where hundreds of people voted which I called "Preset Arena". The results can be found [here](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md).

A key takeaway is that the best presets are:

* **For Instruct**: Divine Intellect, Big O, simple-1, Space Alien, StarChat, Titanic, tfs-with-top-a, Asterism, Contrastive Search (only works for the Transformers loader at the moment).
* **For Chat**: Midnight Enigma, Yara, Shortwave.

The other presets are:

* Mirostat: a special decoding technique first implemented in llama.cpp and then adapted into this repository for all loaders. Many people have obtained positive results with it for chat.
* LLaMA-Precise: a legacy preset that was the default for the web UI before the Preset Arena.
* Debug-deterministic: disables sampling. It is useful for debugging, or if you intentionally want to use greedy decoding.

#### Parameters description

For a technical description of the parameters, the [transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) is a good reference.

* **Temperature**: Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.
* **top_p**: If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.
* **top_k**: Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.
* **repetition_penalty**: Penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.
* **repetition_penalty_range**: The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used.
* **typical_p**: If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.
* **tfs**: Tries to detect a tail of low probability tokens in the distribution and disconsiders those tokens. See [this blog post](https://www.trentonbricken.com/Tail-Free-Sampling/) for details. The closer to 0, the more discarded tokens.
* **top_a**: Tokens with probability smaller than `(top_a) * (probability of the most likely token)^2` are discarded.
* **epsilon_cutoff**: In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled.
* **eta_cutoff**: In units of 1e-4; a reasonable value is 3. The main parameter of the special Eta Sampling technique. See [this paper](https://arxiv.org/pdf/2210.15191.pdf) for a description.


* **encoder_repetition_penalty**: Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.
* **no_repeat_ngram_size**: If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.
* **min_length**: Minimum generation length in tokens.
* **penalty_alpha**: Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.

