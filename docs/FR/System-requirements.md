Voici les exigences en VRAM et RAM (en MiB) pour exécuter certains exemples de modèles **en précision 16 bits (par défaut)** :

| modèle                 |   VRAM (GPU) |     RAM |
|:-----------------------|-------------:|--------:|
| arxiv_ai_gpt2          |      1512.37 | 5824.2  |
| blenderbot-1B-distill  |      2441.75 | 4425.91 |
| opt-1.3b               |      2509.61 | 4427.79 |
| gpt-neo-1.3b           |      2605.27 | 5851.58 |
| opt-2.7b               |      5058.05 | 4863.95 |
| gpt4chan_model_float16 |     11653.7  | 4437.71 |
| gpt-j-6B               |     11653.7  | 5633.79 |
| galactica-6.7b         |     12697.9  | 4429.89 |
| opt-6.7b               |     12700    | 4368.66 |
| bloomz-7b1-p3          |     13483.1  | 4470.34 |

#### Mode GPU avec précision 8 bits

Permet de charger des modèles qui ne rentreraient normalement pas dans votre GPU. Activé par défaut pour les modèles 13b et 20b dans cette interface web.

| modèle         |   VRAM (GPU) |     RAM |
|:---------------|-------------:|--------:|
| opt-13b        |      12528.1 | 1152.39 |
| gpt-neox-20b   |      20384   | 2291.7  |

#### Mode CPU (précision 32 bits)

Beaucoup plus lent, mais ne nécessite pas de GPU.

Sur mon i5-12400F, les modèles 6B prennent environ 10-20 secondes pour répondre en mode chat, et environ 5 minutes pour générer une complétion de 200 jetons.

| modèle                 |      RAM |
|:-----------------------|---------:|
| arxiv_ai_gpt2          |  4430.82 |
| gpt-neo-1.3b           |  6089.31 |
| opt-1.3b               |  8411.12 |
| blenderbot-1B-distill  |  8508.16 |
| opt-2.7b               | 14969.3  |
| bloomz-7b1-p3          | 21371.2  |
| gpt-j-6B               | 24200.3  |
| gpt4chan_model         | 24246.3  |
| galactica-6.7b         | 26561.4  |
| opt-6.7b               | 29596.6  |