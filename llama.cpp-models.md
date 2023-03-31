Available after [#447](https://github.com/oobabooga/text-generation-webui/pull/447) thanks to [@thomasantony](https://github.com/thomasantony) and his [llamacpp-python](https://github.com/thomasantony/llamacpp-python) library.

## Using llama.cpp in the web UI

1. Re-install the requirements to get `llamacpp` intalled:

```
pip install -r requirements.txt --upgrade
```

2. Follow the instructions in the llama.cpp README to generate the `ggml-model-q4_0.bin` file: https://github.com/ggerganov/llama.cpp#usage

3. Create a folder with name starting in `llamacpp` inside `models/`. For instance, `models/llamacpp-7b`

4. Place `ggml-model-q4_0.bin` inside that folder.

5. Start the web UI normally:

```
python server.py --model llamacpp-7b
```