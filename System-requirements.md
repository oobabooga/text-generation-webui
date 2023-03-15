These are the VRAM and RAM requirements (in MiB) to run some examples of models **in 16-bit (default) precision**:

| model                  |   VRAM (GPU) |     RAM |
|:-----------------------|-------------:|--------:|
| pygmalion-350m         |       632.61 | 2671.39 |
| OPT-350M-Erebus        |       632.61 | 4366.4  |
| arxiv_ai_gpt2          |      1512.37 | 5824.2  |
| blenderbot-1B-distill  |      2441.75 | 4425.91 |
| opt-1.3b               |      2509.61 | 4427.79 |
| gpt-neo-1.3b           |      2605.27 | 5851.58 |
| pygmalion-1.3b         |      2794.25 | 2512.87 |
| opt-2.7b               |      5058.05 | 4863.95 |
| pygmalion-2.7b         |      5185.59 | 1963.48 |
| pygmalion-6b           |     11653.7  | 2605.79 |
| gpt4chan_model_float16 |     11653.7  | 4437.71 |
| gpt-j-6B               |     11653.7  | 5633.79 |
| galactica-6.7b         |     12697.9  | 4429.89 |
| OPT-6.7B-Erebus        |     12700    | 4388.55 |
| opt-6.7b               |     12700    | 4368.66 |
| bloomz-7b1-p3          |     13483.1  | 4470.34 |

#### GPU mode with 8-bit precision

Allows you to load models that would not normally fit into your GPU. Enabled by default for 13b and 20b models in this web UI.

| model          |   VRAM (GPU) |     RAM |
|:---------------|-------------:|--------:|
| OPT-13B-Erebus |      12528   | 1819.41 |
| opt-13b        |      12528.1 | 1152.39 |
| gpt-neox-20b   |      20384   | 2291.7  |

#### CPU mode (32-bit precision)

A lot slower, but does not require a GPU. 

On my i5-12400F, 6B models take around 10-20 seconds to respond in chat mode, and around 5 minutes to generate a 200 tokens completion. 

| model                  |      RAM |
|:-----------------------|---------:|
| pygmalion-350m         |  2092    |
| OPT-350M-Erebus        |  4372.19 |
| arxiv_ai_gpt2          |  4430.82 |
| gpt-neo-1.3b           |  6089.31 |
| pygmalion-1.3b         |  7838.89 |
| opt-1.3b               |  8411.12 |
| blenderbot-1B-distill  |  8508.16 |
| pygmalion-2.7b         | 11153.8  |
| opt-2.7b               | 14969.3  |
| bloomz-7b1-p3          | 21371.2  |
| gpt-j-6B               | 24200.3  |
| gpt4chan_model         | 24246.3  |
| pygmalion-6b           | 24933.1  |
| OPT-6.7B-Erebus        | 25656.8  |
| galactica-6.7b         | 26561.4  |
| opt-6.7b               | 29596.6  |
