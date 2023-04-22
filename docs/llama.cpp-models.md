## Using llama.cpp in the web UI

1. Re-install the requirements.txt:

```
pip install -r requirements.txt -U
```

2. Follow the instructions in the llama.cpp README to generate the `ggml-model-q4_0.bin` file: https://github.com/ggerganov/llama.cpp#usage

3. Create a folder inside `models/` for your model and put `ggml-model-q4_0.bin` in it. For instance, `models/llamacpp-7b/ggml-model-q4_0.bin`.

4. Start the web UI normally:

```
python server.py --model llamacpp-7b
```

* This procedure should work for any `ggml*.bin` file. Just put it in a folder, and use the name of this folder as the argument after `--model` or as the model loaded inside the interface.
* You can change the number of threads with `--threads N`.

## Performance

This was the performance of llama-7b int4 on my i5-12400F:

> Output generated in 33.07 seconds (6.05 tokens/s, 200 tokens, context 17)

## Limitations

~* The parameter sliders in the interface (temperature, top_p, top_k, etc) are completely ignored. So only the default parameters in llama.cpp can be used.~

~* Only 512 tokens of context can be used.~

~Both of these should be improved soon when llamacpp-python receives an update.~

