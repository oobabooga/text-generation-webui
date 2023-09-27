# Generation Parameters

For a technical description of the parameters, the [transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) is a good reference.

The best presets, according to the [Preset Arena](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md) experiment, are:

**Instruction following:**

1) Divine Intellect
2) Big O
3) simple-1
4) Space Alien
5) StarChat
6) Titanic
7) tfs-with-top-a
8) Asterism
9) Contrastive Search

**Chat:**

1) Midnight Enigma
2) Yara
3) Shortwave

### Temperature

Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.

### top_p

If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.

### top_k

Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.

### typical_p

If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.

### epsilon_cutoff

In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled. Should be used with top_p, top_k, and eta_cutoff set to 0.

### eta_cutoff

In units of 1e-4; a reasonable value is 3. Should be used with top_p, top_k, and epsilon_cutoff set to 0.

### repetition_penalty

Exponential penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.

### repetition_penalty_range

The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used.

### encoder_repetition_penalty

Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.

### no_repeat_ngram_size

If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.

### min_length

Minimum generation length in tokens.

### penalty_alpha

Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.
