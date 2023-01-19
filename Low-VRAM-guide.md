If you GPU is not large enough to fit a model, try these in the following order:

### Load the model in 8-bit mode
`python server.py --load-in-8bit`

This reduces the memory usage by half with no noticeable loss in performance. Only newer GPUs support 8-bit mode.

### Split the model across your GPU and CPU

`python server.py --auto-devices`

If you can load the model with this command but it runs out of memory when you try to generate text, try limiting the amount of memory allocated to the GPU: 

`python server.py --auto-devices --max-gpu-memory 10`

where the number is in GiB.

### Send layers to a disk cache

As a desperate last measure, you can split the model across your GPU, CPU, and disk:

`python server.py --auto-devices --disk`

With this, I am able to load a 30b model into my RTX 3090, but it takes 10 seconds to generate 1 word.