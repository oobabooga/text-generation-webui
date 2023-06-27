# Using starcoder.cpp in the web UI

## Setting up the models

#### Pre-converted

Place the model in the `models` folder, making sure that its name
contains `starcoder` or `starchat` in the beginning, `ggml` somewhere
in the middle and ends in `.bin`.

You can find converted models here:

- [StarChat Alpha](https://huggingface.co/NeoDim/starchat-alpha-GGML)
- [StarCoder](https://huggingface.co/NeoDim/starcoder-GGML)
- [StarCoderBase](https://huggingface.co/NeoDim/starcoderbase-GGML)

#### Convert models yourself

Follow the instructions
[here](https://github.com/ggerganov/ggml/tree/master/examples/starcoder)

There is also
[starcoder.cpp](https://github.com/bigcode-project/starcoder.cpp#quantizing-the-models)
but there is an [issue](https://github.com/bigcode-project/starcoder.cpp/issues/11)


