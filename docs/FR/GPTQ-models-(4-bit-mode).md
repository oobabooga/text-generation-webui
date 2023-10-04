GPTQ est un algorithme de quantification intelligent qui réoptimise légèrement les poids pendant la quantification afin que la perte de précision soit compensée par rapport à une quantification arrondie au plus près. Pour plus de détails, consultez l'article : https://arxiv.org/abs/2210.17323

Les modèles GPTQ 4 bits réduisent l'utilisation de la VRAM d'environ 75%. Ainsi, LLaMA-7B peut être utilisé sur un GPU de 6 Go, et LLaMA-30B sur un GPU de 24 Go.

## Aperçu

Il existe actuellement deux façons de charger les modèles GPTQ dans l'interface web :

* En utilisant AutoGPTQ :
  * Prend en charge plus de modèles
  * Standardisé (pas besoin de deviner un quelconque paramètre)
  * Est une véritable bibliothèque Python
  * ~Aucun fichier wheel n'est actuellement disponible, donc cela nécessite une compilation manuelle~
  * Prend en charge le chargement des modèles triton et cuda

* En utilisant GPTQ-for-LLaMa directement :
  * Délestage CPU plus rapide
  * Inférence multi-GPU plus rapide
  * Prend en charge le chargement des LoRAs via un monkey patch
  * Vous devez manuellement déterminer les paramètres wbits/groupsize/model_type pour pouvoir charger le modèle
  * Prend en charge soit uniquement cuda, soit uniquement triton selon la branche

Pour créer de nouvelles quantifications, je recommande d'utiliser AutoGPTQ : https://github.com/PanQiWei/AutoGPTQ

## AutoGPTQ

### Installation

Aucune étape supplémentaire n'est nécessaire car AutoGPTQ est déjà dans le fichier `requirements.txt` pour le webui. Si vous souhaitez ou devez toujours l'installer manuellement pour une raison quelconque, voici les commandes :

```
conda activate textgen
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .
```

La dernière commande nécessite que `nvcc` soit installé (voir les [instructions ci-dessus](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#step-1-install-nvcc)).

### Utilisation

Lorsque vous quantifiez un modèle avec AutoGPTQ, un dossier contenant un fichier appelé `quantize_config.json` sera généré. Placez ce dossier dans votre dossier `models/` et chargez-le avec le flag `--autogptq` :

```
python server.py --autogptq --model nom_du_modèle
```

Ou cochez la case `autogptq` dans l'onglet "Modèle" de l'UI avant de charger le modèle.

### Délestage (Offloading)

Pour effectuer un délestage CPU ou une inférence multi-GPU avec AutoGPTQ, utilisez le flag `--gpu-memory`. C'est actuellement un peu plus lent que le délestage avec l'option `--pre_layer` dans GPTQ-for-LLaMa.

Pour le délestage CPU :

```
python server.py --autogptq --gpu-memory 3000MiB --model nom_du_modèle
```

Pour l'inférence multi-GPU :

```
python server.py --autogptq --gpu-memory 3000MiB 6000MiB --model nom_du_modèle
```

### Utiliser LoRAs avec AutoGPTQ

Fonctionne bien pour un seul LoRA.

## GPTQ-for-LLaMa

GPTQ-for-LLaMa est l'adaptation originale de GPTQ pour le modèle LLaMA. Cela a été rendu possible par [@qwopqwop200](https://github.com/qwopqwop200/GPTQ-for-LLaMa): https://github.com/qwopqwop200/GPTQ-for-LLaMa

Un package Python contenant les deux principales versions CUDA de GPTQ-for-LLaMa est utilisé pour simplifier l'installation et la compatibilité: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA

### Roues précompilées (Precompiled wheels)

Généreusement fournies par notre ami jllllll: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA/releases

Les roues sont incluses dans requirements.txt et sont installées avec le webui sur les systèmes pris en charge.

### Installation manuelle

#### Étape 1 : installez nvcc

```
conda activate textgen
conda install cuda -c nvidia/label/cuda-11.7.1
```

La commande ci-dessus prend environ 10 minutes à exécuter et ne montre aucune barre de progression ou mise à jour en cours.

Vous devrez également avoir un compilateur C++ installé. Sur Linux, `sudo apt install build-essential` ou équivalent suffit. Sur Windows, Visual Studio ou Visual Studio Build Tools est requis.

Si vous utilisez une version plus ancienne de CUDA toolkit (par exemple 11.7) mais la dernière version de `gcc` et `g++` (12.0+) sur Linux, vous devriez rétrograder avec : `conda install -c conda-forge gxx==11.3.0`. La compilation du noyau échouera sinon.

#### Étape 2 : compilez les extensions CUDA

```
python -m pip install git+https://github.com/jllllll/GPTQ-for-LLaMa-CUDA -v
```

### Obtenir les poids LLaMA pré-convertis
Bien sûr, voici la traduction:

---

* Téléchargement direct (recommandé):

https://huggingface.co/Neko-Institute-of-Science/LLaMA-7B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-13B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-30B-4bit-128g

https://huggingface.co/Neko-Institute-of-Science/LLaMA-65B-4bit-128g

Ces modèles ont été convertis avec `desc_act=True`. Ils fonctionnent parfaitement avec ExLlama. Pour AutoGPTQ, ils ne fonctionneront que sur Linux avec l'option `triton` activée.

* Torrent:

https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617

https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105

Ces modèles ont été convertis avec `desc_act=False`. De ce fait, ils sont moins précis, mais ils fonctionnent avec AutoGPTQ sous Windows. Les versions `128g` sont meilleures à partir de 13b, mais moins bonnes pour 7b. Les fichiers de tokenisation dans les torrents sont obsolètes, en particulier les fichiers nommés `tokenizer_config.json` et `special_tokens_map.json`. Vous pouvez trouver ces fichiers ici : https://huggingface.co/oobabooga/llama-tokenizer

### Démarrage de l'interface web:

Utilisez le drapeau `--gptq-for-llama`.

Pour les modèles convertis sans `group-size` :

```
python server.py --model llama-7b-4bit --gptq-for-llama 
```

Pour les modèles convertis avec `group-size`:

```
python server.py --model llama-13b-4bit-128g  --gptq-for-llama --wbits 4 --groupsize 128
```

Les drapeaux en ligne de commande `--wbits` et `--groupsize` sont automatiquement détectés en fonction des noms des dossiers dans de nombreux cas.

### Offloading CPU

Il est possible de transférer une partie des couches du modèle 4 bits vers le CPU avec le drapeau `--pre_layer`. Plus le nombre après `--pre_layer` est élevé, plus de couches seront allouées au GPU.

Avec cette commande, je peux exécuter llama-7b avec 4GB de VRAM:

```
python server.py --model llama-7b-4bit --pre_layer 20
```

Voici les performances:

```
Sortie générée en 123.79 secondes (1.61 tokens/s, 199 tokens)
```

Vous pouvez également utiliser plusieurs GPU avec `pre_layer` si vous utilisez le fork oobabooga de GPTQ, par exemple `--pre_layer 30 60` chargera un modèle LLaMA-30B moitié sur votre premier GPU et moitié sur votre second, ou `--pre_layer 20 40` chargera 20 couches sur GPU-0, 20 couches sur GPU-1, et 20 couches transférées vers le CPU.

### Utilisation de LoRAs avec GPTQ-for-LLaMa

Ceci nécessite l'utilisation d'un monkey patch pris en charge par cette interface web : https://github.com/johnsmith0031/alpaca_lora_4bit

Pour l'utiliser :

1. Installez alpaca_lora_4bit en utilisant pip

```
git clone https://github.com/johnsmith0031/alpaca_lora_4bit.git
cd alpaca_lora_4bit
git fetch origin winglian-setup_pip
git checkout winglian-setup_pip
pip install .
```

2. Démarrez l'interface avec le drapeau `--monkey-patch`:

```
python server.py --model llama-7b-4bit-128g --listen --lora tloen_alpaca-lora-7b --monkey-patch
```
