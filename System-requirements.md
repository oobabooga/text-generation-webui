These are the VRAM (in GiB) and RAM (in MiB) requirements to run some examples of models.

#### GPU mode (default)

| model                  |   VRAM (GPU) |     RAM |
|:-----------------------|-------------:|--------:|
| OPT-350M-Erebus        |         0.62 | 1939.3  |
| arxiv_ai_gpt2          |         1.48 | 6350.68 |
| blenderbot-1B-distill  |         2.38 | 2705.9  |
| opt-1.3b               |         2.45 | 2868.12 |
| gpt-neo-1.3b           |         2.54 | 4047.04 |
| opt-2.7b               |         4.94 | 4220.01 |
| gpt4chan_model_float16 |        11.38 | 1909.79 |
| gpt-j-6B               |        11.38 | 3959.55 |
| galactica-6.7b         |        12.4  | 1933.19 |
| opt-6.7b               |        12.4  | 1944.21 |
| bloomz-7b1-p3          |        13.17 | 1845.58 |

#### GPU mode with 8-bit precision

Allows you to load models that would not normally fit into your GPU. Enabled by default for 13b and 20b models in this webui.

| model          |   VRAM (GPU) |     RAM |
|:---------------|-------------:|--------:|
| OPT-13B-Erebus |        12.23 |  749.08 |
| opt-13b        |        12.23 | 1258.95 |
| gpt-neox-20b   |        19.91 | 2104.04 |

#### CPU mode

A lot slower, but does not require a GPU.

| model                  |      RAM |
|:-----------------------|---------:|
| OPT-350M-Erebus        |  2622.17 |
| arxiv_ai_gpt2          |  3764.81 |
| gpt-neo-1.3b           |  5937.81 |
| opt-1.3b               |  7346.08 |
| blenderbot-1B-distill  |  7565.36 |
| opt-2.7b               | 12527.31 |
| bloomz-7b1-p3          | 23613.9  |
| gpt-j-6B               | 23975.5  |
| gpt4chan_model         | 23999.5  |
| galactica-6.7b         | 26248    |
| opt-6.7b               | 27334.2  |
