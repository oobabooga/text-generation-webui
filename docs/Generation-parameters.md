# Generation parameters

For a description of the generation parameters provided by the transformers library, see this link: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

### llama.cpp

llama.cpp only uses the following parameters:

* temperature
* top_p
* top_k
* repetition_penalty
* tfs
* mirostat_mode
* mirostat_tau
* mirostat_eta

### ExLlama

ExLlama only uses the following parameters:

* temperature
* top_p
* top_k
* repetition_penalty
* repetition_penalty_range
* typical_p

### RWKV

RWKV only uses the following parameters when loaded through the old .pth weights:

* temperature
* top_p
* top_k
