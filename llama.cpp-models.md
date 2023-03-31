Available after [#447](https://github.com/oobabooga/text-generation-webui/pull/447) thanks to [@thomasantony](https://github.com/thomasantony) and his [llamacpp-python](https://github.com/thomasantony/llamacpp-python) library.

## Using llama.cpp in the web UI

1. Re-install the requirements to get `llamacpp` intalled:

```
pip install -r requirements.txt --upgrade
```

2. Follow the instructions in the llama.cpp README to generate the `ggml-model-q4_0.bin` file: https://github.com/ggerganov/llama.cpp#usage

3. Create a folder inside `models/` for your model and put `ggml-model-q4_0.bin` in it. For instance, `models/llamacpp-7b/ggml-model-q4_0.bin`.

4. Start the web UI normally:

```
python server.py --model llamacpp-7b
```

**This procedure should work for any `ggml*.bin` file.**

## Performance

This was the performance of llama-7b int4 on my i5-12400F:

> Output generated in 44.10 seconds (4.53 tokens/s, 200 tokens)
