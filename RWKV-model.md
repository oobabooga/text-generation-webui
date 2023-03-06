> RWKV: RNN with Transformer-level LLM Performance
>
> It combines the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).

https://github.com/BlinkDL/RWKV-LM

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

And also put it directly in the `models` folder. Make sure to not rename it. It should be called `20B_tokenizer.json`.

[20B_tokenizer.json](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/20B_tokenizer.json)

#### 3. Launch the web UI

As you would with any other model, for instance with

```
python server.py --listen  --no-stream --model RWKV-4-Pile-169M-20220807-8023.pth
```

#### Setting a custom strategy

It is possible to control the offloading strategy for the model with the `--rwkv-strategy` flag. Possible values include:

```
"cpu fp32"
"cuda fp16"
"cuda fp16 *30 -> cpu fp32"
```

For instance,

```
python server.py --listen  --no-stream --rwkv-strategy "cuda fp16"
```

will run the model in the CPU, whereas

```
python server.py --listen  --no-stream --rwkv-strategy "cuda fp16 *30 -> cpu fp32"
```

will split the layers across the CPU and GPU. The higher the number (in this case, `30`), the more memory will be allocated to the GPU.

#### Compiling the CUDA kernel

WIP
