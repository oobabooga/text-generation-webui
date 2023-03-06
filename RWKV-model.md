> RWKV: RNN with Transformer-level LLM Performance
>
> It combines the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).

https://github.com/BlinkDL/RWKV-LM

https://github.com/BlinkDL/ChatRWKV

## Using RWKV in the web UI

#### 1. Download the model

It is available in different sizes:

* https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth
* https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-20230109-ctx4096.pth
* https://huggingface.co/BlinkDL/rwkv-4-pile-14b/resolve/main/RWKV-4-Pile-14B-20230228-ctx4096-test663.pth

There are also older releases with smaller sizes like:

* https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth

Download the chosen .pth and put it directly in the `models` folder. 

#### 2. Download the tokenizer

[20B_tokenizer.json](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/20B_tokenizer.json)

Also put it directly in the `models` folder. Make sure to not rename it. It should be called `20B_tokenizer.json`.

#### 3. Launch the web UI

As you would with any other model, for instance, with

```
python server.py --listen  --no-stream --model RWKV-4-Pile-169M-20220807-8023.pth
```

## Setting a custom strategy

It is possible to control the offloading strategy for the model with the `--rwkv-strategy` flag. Possible values include:

```
"cpu fp32" # CPU mode
"cuda fp16" # GPU mode with float16 precision
"cuda fp16 *30 -> cpu fp32" # GPU+CPU offloading. The higher the number after *, the higher the GPU allocation.
"cuda fp16i8" # GPU mode with 8-bit precision
```

See the README of the PyPl package for more details: https://pypi.org/project/rwkv/

## Compiling the CUDA kernel

You can compile the CUDA kernel for the model with the `--rwkv-cuda-on` flag. This should improve the performance a lot but I haven't be able to get it to work yet.