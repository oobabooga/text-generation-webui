## Using llama.cpp in the web UI

#### Pre-converted models

Simply place the model in the `models` folder, making sure that its name contains `ggml` somewhere and ends in `.bin`.

#### Convert LLaMA yourself

Follow the instructions in the llama.cpp README to generate the `ggml-model-q4_0.bin` file: https://github.com/ggerganov/llama.cpp#usage

## Performance

This was the performance of llama-7b int4 on my i5-12400F:

> Output generated in 33.07 seconds (6.05 tokens/s, 200 tokens, context 17)

You can change the number of threads with `--threads N`.
