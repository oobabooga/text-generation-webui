You have now entered a hidden corner of the internet.

A confusing yet intriguing realm of paradoxes and contradictions.

A place where you will find out that what you thought you knew, you in fact didn't know, and what you didn't know was in front of you all along.

![](https://i.pinimg.com/originals/6e/e2/7b/6ee27bad351d3aca470d80f1033ba9c6.jpg)

*In other words, here I will document little-known facts about this web UI that I could not find another place for in the wiki.*

#### You can train LoRAs in CPU mode

Load the web UI with

```
python server.py --cpu
```

and start training the LoRA from the training tab as usual.

#### 8-bit mode works with CPU offloading

```
python server.py --load-in-8bit --gpu-memory 4000MiB
```

#### `--pre_layer`, and not `--gpu-memory`, is the right way to do CPU offloading with 4-bit models

```
python server.py --wbits 4 --groupsize 128 --pre_layer 20
```

#### Models can be loaded in 32-bit, 16-bit, 8-bit, and 4-bit modes

```
python server.py --cpu
python server.py
python server.py --load-in-8bit
python server.py --wbits 4
```

#### Instruction-following templates are represented as chat characters

https://github.com/oobabooga/text-generation-webui/tree/main/characters/instruction-following

#### The right way to run Alpaca, Open Assistant, Vicuna, etc is Instruct mode, not normal chat mode

Otherwise the prompt will not be formatted correctly.

1. Start the web UI with

```
python server.py --chat
```

2. Click on the "instruct" option under "Chat modes"

3. Select the correct template in the hidden dropdown menu that will become visible. 

#### Notebook mode is best mode

Ascended individuals have realized that notebook mode is the superset of chat mode and can do chats with ultimate flexibility, including group chats, editing replies, starting a new bot reply in a given way, and impersonating.

#### RWKV is a RNN 

Most models are transformers, but not RWKV, which is a RNN. It's a great model.

#### `--gpu-memory` is not a hard limit on the GPU memory

It is simply a parameter that is passed to the `accelerate` library while loading the model. More memory will be allocated during generation. That's why this parameter has to be set to less than your total GPU memory.

#### Contrastive search perhaps the best preset

But it uses a ton of VRAM.

#### You can check the sha256sum of downloaded models with the download script

```
python download-model.py facebook/galactica-125m --check
```

#### The download script continues interrupted downloads by default

It doesn't start over.

#### You can download models with multiple threads

```
python download-model.py facebook/galactica-125m --threads 8
```
