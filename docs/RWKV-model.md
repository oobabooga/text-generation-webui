> RWKV: RNN with Transformer-level LLM Performance
>
> It combines the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).

https://github.com/BlinkDL/RWKV-LM

https://github.com/BlinkDL/ChatRWKV

## Using RWKV in the web UI

### Hugging Face weights

Simply download the weights from https://huggingface.co/RWKV and load them as you would for any other model.

There is a bug in transformers==4.29.2 that prevents RWKV from being loaded in 8-bit mode. You can install the dev branch to solve this bug: `pip install git+https://github.com/huggingface/transformers`

### Original .pth weights

The instructions below are from before RWKV was supported in transformers, and they are kept for legacy purposes. The old implementation is possibly faster, but it lacks the full range of samplers that the transformers library offers.

#### 0. Install the RWKV library

```
pip install rwkv
```

`0.7.3` was the last version that I tested. If you experience any issues, try ```pip install rwkv==0.7.3```.

#### 1. Download the model

It is available in different sizes:

* https://huggingface.co/BlinkDL/rwkv-4-pile-3b/
* https://huggingface.co/BlinkDL/rwkv-4-pile-7b/
* https://huggingface.co/BlinkDL/rwkv-4-pile-14b/

There are also older releases with smaller sizes like:

* https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth

Download the chosen `.pth` and put it directly in the `models` folder. 

#### 2. Download the tokenizer

[20B_tokenizer.json](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/20B_tokenizer.json)

Also put it directly in the `models` folder. Make sure to not rename it. It should be called `20B_tokenizer.json`.

#### 3. Launch the web UI

No additional steps are required. Just launch it as you would with any other model.

```
python server.py --listen  --no-stream --model RWKV-4-Pile-169M-20220807-8023.pth
```

#### Setting a custom strategy

It is possible to have very fine control over the offloading and precision for the model with the `--rwkv-strategy` flag. Possible values include:

```
"cpu fp32" # CPU mode
"cuda fp16" # GPU mode with float16 precision
"cuda fp16 *30 -> cpu fp32" # GPU+CPU offloading. The higher the number after *, the higher the GPU allocation.
"cuda fp16i8" # GPU mode with 8-bit precision
```

See the README for the PyPl package for more details: https://pypi.org/project/rwkv/

#### Compiling the CUDA kernel

You can compile the CUDA kernel for the model with `--rwkv-cuda-on`. This should improve the performance a lot but I haven't been able to get it to work yet.
