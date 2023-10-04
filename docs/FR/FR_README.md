

# Text generation WebUI

Une interface web Gradio pour les grands modèles de langage.

Son objectif est de devenir le [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) de la génération de texte.

|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_instruct.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_chat.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_default.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_parameters.png) |


### Sommaire

- [Installation](#Installation)
- [Télécharger les modèles](#Télécharger-les-modèles)
- [Démarrer l'interface web](#Démarrer-l'interface-web)
- [Préréglages](#Préréglages)
- [Contribuer](#Contribuer)
- [Communauté](#Communauté)
- [Remerciements](#Remerciements)

## Caractéristiques

* 3 modes d'interface : par défaut (deux colonnes), notebook, et chat
* Plusieurs backends de modèles : [transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [ExLlama](https://github.com/turboderp/exllama), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [CTransformers](https://github.com/marella/ctransformers)
* Menu déroulant pour passer rapidement entre différents modèles
* LoRA : charger et décharger les LoRAs à la volée, entraîner un nouveau LoRA avec QLoRA
* Modèles précis d'instructions pour le mode chat, incluant Llama-2-chat, Alpaca, Vicuna, WizardLM, StableLM, et bien d'autres
* Inférence à 4-bit, 8-bit, et CPU via la bibliothèque transformers
* Utilisation des modèles llama.cpp avec les échantillonneurs transformers (`llamacpp_HF` loader)
* [Pipelines multimodaux, incluant LLaVA et MiniGPT-4](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal)
* [Cadre d'extensions](Extensions.md)
* [Personnages de chat personnalisés](Chat-mode.md)
* Streaming de texte très efficace
* Sortie en Markdown avec rendu LaTeX, à utiliser par exemple avec [GALACTICA](https://github.com/paperswithcode/galai)
* API, y compris des points de terminaison pour le streaming websocket ([voir les exemples](https://github.com/oobabooga/text-generation-webui/blob/main/api-examples))

Pour apprendre comment utiliser les différentes fonctionnalités, consultez la documentation : https://github.com/oobabooga/text-generation-webui/tree/main/docs

## Installation

### Installateurs en un clic

1) Clonez ou téléchargez le dépôt.
2) Exécutez le script `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, ou `start_wsl.bat` selon votre système d'exploitation.
3) Sélectionnez votre fournisseur de GPU lorsque cela vous est demandé.
4) Amusez-vous !

#### Comment ça fonctionne

Le script crée un dossier appelé `installer_files` où il configure un environnement Conda utilisant Miniconda. L'installation est autonome : si vous souhaitez réinstaller, il vous suffit de supprimer `installer_files` et d'exécuter à nouveau le script de démarrage.

Pour lancer l'interface web à l'avenir après son installation, exécutez le même script `start`.

#### Mises à jour

Exécutez `update_linux.sh`, `update_windows.bat`, `update_macos.sh`, ou `update_wsl.bat`.

#### Exécution de commandes

Si vous avez besoin d'installer manuellement quelque chose dans l'environnement `installer_files`, vous pouvez lancer un shell interactif à l'aide du script cmd : `cmd_linux.sh`, `cmd_windows.bat`, `cmd_macos.sh`, ou `cmd_wsl.bat`.

#### Définition des flags de ligne de commande

Pour définir des flags de ligne de commande persistants comme `--listen` ou `--api`, éditez le fichier `CMD_FLAGS.txt` avec un éditeur de texte et ajoutez-les. Les flags peuvent également être fournis directement aux scripts de démarrage, par exemple, `./start-linux.sh --listen`.

#### Autres infos

* Il n'est pas nécessaire d'exécuter ces scripts en tant qu'administrateur/root.
* Pour des instructions supplémentaires sur la configuration AMD, la configuration WSL, et l'installation de nvcc, consultez [cette page](One-Click-Installers.md).
* L'installateur a été principalement testé sur les GPU NVIDIA. Si vous trouvez un moyen de l'améliorer pour votre GPU AMD/Intel Arc/Mac Metal, vous êtes fortement encouragé à soumettre une PR à ce dépôt. Le fichier principal à modifier est `one_click.py`.
* Pour une installation automatisée, vous pouvez utiliser les variables d'environnement `GPU_CHOICE`, `LAUNCH_AFTER_INSTALL`, et `INSTALL_EXTENSIONS`. Par exemple : `GPU_CHOICE=A LAUNCH_AFTER_INSTALL=False INSTALL_EXTENSIONS=False ./start_linux.sh`.

### Installation manuelle avec Conda

Recommandé si vous avez une certaine expérience avec la ligne de commande.

#### 0. Installez Conda

https://docs.conda.io/en/latest/miniconda.html

Sur Linux ou WSL, il peut être automatiquement installé avec ces deux commandes ([source](https://educe-ubc.github.io/conda.html)) :

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1. Créez un nouvel environnement conda

```
conda create -n textgen python=3.10.9
conda activate textgen
```


#### 2. Installer Pytorch

| Système | GPU | Commande |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch torchvision torchaudio` |
| Linux/WSL | CPU uniquement | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2` |
| MacOS + MPS | N'importe lequel | `pip3 install torch torchvision torchaudio` |
| Windows | NVIDIA | `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117` |
| Windows | CPU uniquement | `pip3 install torch torchvision torchaudio` |

Les commandes mises à jour peuvent être trouvées ici : https://pytorch.org/get-started/locally/. 

#### 3. Installer l'interface web 

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

#### AMD, Metal, Intel Arc, et CPUs sans AVX2

1) Remplacer la dernière commande ci-dessus par

```
pip install -r requirements_nowheels.txt
```

2) Installer manuellement llama-cpp-python en utilisant la commande appropriée pour votre matériel : [Installation depuis PyPI](https://github.com/abetlen/llama-cpp-python#installation-from-pypi).

3) Faire de même pour CTransformers : [Installation](https://github.com/marella/ctransformers#installation).

4) AMD : Installer manuellement AutoGPTQ : [Installation](https://github.com/PanQiWei/AutoGPTQ#installation).

5) AMD : Installer manuellement [ExLlama](https://github.com/turboderp/exllama) en le clonant simplement dans le dossier `repositories` (il sera automatiquement compilé à l'exécution par la suite) :

```
cd text-generation-webui
git clone https://github.com/turboderp/exllama repositories/exllama
```

#### bitsandbytes sur les anciens GPU NVIDIA

bitsandbytes >= 0.39 peut ne pas fonctionner. Dans ce cas, pour utiliser `--load-in-8bit`, vous devrez peut-être rétrograder comme ceci :

* Linux : `pip install bitsandbytes==0.38.1`
* Windows : `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`


### Alternative : Docker

```
ln -s docker/{Dockerfile,docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
# Éditez .env et définissez TORCH_CUDA_ARCH_LIST en fonction de votre modèle de GPU
docker compose up --build
```

* Vous devez avoir docker compose v2.17 ou une version supérieure installée. Consultez [ce guide](Docker.md) pour les instructions.
* Pour des fichiers docker supplémentaires, consultez [ce dépôt](https://github.com/Atinoda/text-generation-webui-docker).

### Mettre à jour les exigences

De temps en temps, le `requirements.txt` change. Pour mettre à jour, utilisez ces commandes :

```
conda activate textgen
cd text-generation-webui
pip install -r requirements.txt --upgrade
```

## Télécharger les modèles
[Retour au sommaire](#Sommaire)

Les modèles doivent être placés dans le dossier `text-generation-webui/models`. Ils sont généralement téléchargés depuis [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* Les modèles Transformers ou GPTQ se composent de plusieurs fichiers et doivent être placés dans un sous-dossier. Exemple :

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

* Les modèles GGUF sont un seul fichier et doivent être placés directement dans `models`. Exemple :

```
text-generation-webui
├── models
│   ├── llama-2-13b-chat.Q4_K_M.gguf
```

Dans les deux cas, vous pouvez utiliser l'onglet "Model" de l'UI pour télécharger le modèle depuis Hugging Face automatiquement. Il est également possible de télécharger via la ligne de commande avec `python download-model.py organization/model` (utilisez `--help` pour voir toutes les options).

#### GPT-4chan

<details>
<summary>
Instructions
</summary>

[GPT-4chan](https://huggingface.co/ykilcher/gpt-4chan) a été retiré de Hugging Face, vous devez donc le télécharger ailleurs. Vous avez deux options :

* Torrent : [16-bit](https://archive.org/details/gpt4chan_model_float16) / [32-bit](https://archive.org/details/gpt4chan_model)
* Téléchargement direct : [16-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model_float16/) / [32-bit](https://theswissbay.ch/pdf/_notpdf_/gpt4chan_model/)

La version 32 bits est pertinente uniquement si vous avez l'intention d'exécuter le modèle en mode CPU. Sinon, vous devriez utiliser la version 16 bits.

Après avoir téléchargé le modèle, suivez ces étapes :

1. Placez les fichiers sous `models/gpt4chan_model_float16` ou `models/gpt4chan_model`.
2. Placez le fichier config.json de GPT-J 6B dans ce même dossier : [config.json](https://huggingface.co/EleutherAI/gpt-j-6B/raw/main/config.json).
3. Téléchargez les fichiers tokenizer de GPT-J 6B (ils seront automatiquement détectés lorsque vous tenterez de charger GPT-4chan) :

```
python download-model.py EleutherAI/gpt-j-6B --text-only
```

Lorsque vous chargez ce modèle en modes par défaut ou notebook, l'onglet "HTML" affichera le texte généré au format 4chan :

![Image3](https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png)

</details>

## Démarrer l'interface web
[Retour au sommaire](#Sommaire)

    conda activate textgen
    cd text-generation-webui
    python server.py

Ensuite, naviguez vers 

`http://localhost:7860/?__theme=dark`


Optionnellement, vous pouvez utiliser les flags de ligne de commande suivants. Si vous rencontrez un problème avec ces commandes, veuillez consulter la [DOCUMENTATION ICI](README.md) avant de signaler un problème.

#### Paramètres de base

| Indicateur                                  | Description |
|---------------------------------------------|-------------|
| `-h`, `--help`                              | Affiche ce message d'aide et quitte. |
| `--multi-user`                              | Mode multi-utilisateurs. Les historiques de chat ne sont pas sauvegardés ni chargés automatiquement. ATTENTION : ceci est hautement expérimental. |
| `--character CHARACTER`                     | Le nom du personnage à charger par défaut en mode chat. |
| `--model MODEL`                             | Nom du modèle à charger par défaut. |
| `--lora LORA [LORA ...]`                    | La liste des LoRAs à charger. Si vous voulez en charger plus d'une, écrivez les noms séparés par des espaces. |
| `--model-dir MODEL_DIR`                     | Chemin vers le répertoire contenant tous les modèles. |
| `--lora-dir LORA_DIR`                       | Chemin vers le répertoire avec tous les loras. |
| `--model-menu`                              | Affiche un menu de modèles dans le terminal lorsque l'interface web est lancée pour la première fois. |
| `--settings SETTINGS_FILE`                  | Charge les paramètres d'interface par défaut depuis ce fichier yaml. Voir `settings-template.yaml` pour un exemple. Si vous créez un fichier appelé `settings.yaml`, ce fichier sera chargé par défaut sans avoir besoin d'utiliser le flag `--settings`. |
| `--extensions EXTENSIONS [EXTENSIONS ...]`  | La liste des extensions à charger. Si vous souhaitez en charger plusieurs, écrivez les noms séparés par des espaces. |
| `--verbose`                                 | Affiche les messages à l'écran. |
| `--chat-buttons`                            | Affiche des boutons sur l'onglet de chat au lieu du menu contextuel. |

#### Chargeur de modèle

| Indicateur                                 | Description |
|--------------------------------------------|-------------|
| `--loader LOADER`                          | Choisissez le chargeur de modèle manuellement, sinon il sera autodétecté. Options valides : transformers, autogptq, gptq-for-llama, exllama, exllama_hf, llamacpp, rwkv, ctransformers |

#### Accelerate/transformers

| Indicateur                                  | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Utilisez le CPU pour générer du texte. Attention : L'entraînement sur CPU est extrêmement lent.|
| `--auto-devices`                            | Répartit automatiquement le modèle entre le(s) GPU(s) disponible(s) et le CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Mémoire GPU maximale en GiB à allouer par GPU. Exemple : `--gpu-memory 10` pour un seul GPU, `--gpu-memory 10 5` pour deux GPU. Vous pouvez également définir des valeurs en MiB comme `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                   | Mémoire CPU maximale en GiB à allouer pour les poids déchargés. Idem que ci-dessus.|
| `--disk`                                    | Si le modèle est trop volumineux pour votre(vos) GPU et CPU combinés, envoyez les couches restantes sur le disque. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Répertoire pour sauvegarder le cache disque. Par défaut à `cache/`. |
| `--load-in-8bit`                            | Charge le modèle avec une précision de 8 bits (en utilisant bitsandbytes).|
| `--bf16`                                    | Charge le modèle avec une précision bfloat16. Nécessite un GPU NVIDIA Ampere. |
| `--no-cache`                                | Définir `use_cache` sur False lors de la génération de texte. Cela réduit un peu l'utilisation de la VRAM au détriment des performances. |
| `--xformers`                                | Utilisez l'attention efficace en mémoire de xformers. Cela devrait augmenter vos tokens/s. |
| `--sdp-attention`                           | Utilise l'attention sdp de torch 2.0. |
| `--trust-remote-code`                       | Définir trust_remote_code=True lors du chargement d'un modèle. Nécessaire pour ChatGLM et Falcon. |
| `--use_fast`                                | Définir use_fast=True lors du chargement d'un tokenizer. |


#### Accelerate 4-bit

⚠️ Nécessite un minimum de capacité de calcul de 7.0 sur Windows pour le moment.

| Indicateur                                | Description |
|-------------------------------------------|-------------|
| `--load-in-4bit`                          | Charge le modèle avec une précision de 4 bits (en utilisant bitsandbytes). |
| `--compute_dtype COMPUTE_DTYPE`           | Type de données de calcul pour 4-bit. Options valides : bfloat16, float16, float32. |
| `--quant_type QUANT_TYPE`                 | quant_type pour 4-bit. Options valides : nf4, fp4. |
| `--use_double_quant`                      | Utilise use_double_quant pour 4-bit. |

#### GGUF (pour llama.cpp et ctransformers)

| Indicateur                           | Description |
|--------------------------------------|-------------|
| `--threads`                          | Nombre de fils d'exécution à utiliser. |
| `--threads-batch THREADS_BATCH`      | Nombre de fils d'exécution à utiliser pour le traitement des lots/sollicitations. |
| `--n_batch`                          | Nombre maximal de tokens d'invite à regrouper lors de l'appel à llama_eval. |
| `--n-gpu-layers N_GPU_LAYERS`        | Nombre de couches à décharger sur le GPU. Fonctionne uniquement si llama-cpp-python a été compilé avec BLAS. Définir cette valeur à 1000000000 pour décharger toutes les couches sur le GPU. |
| `--n_ctx N_CTX`                      | Taille du contexte de sollicitation. |

#### llama.cpp

| Indicateur                           | Description |
|--------------------------------------|-------------|
| `--mul_mat_q`                        | Active les nouveaux noyaux mulmat. |
| `--tensor_split TENSOR_SPLIT`        | Divise le modèle sur plusieurs GPU, liste séparée par des virgules de proportions, par ex. 18,17. |
| `--llama_cpp_seed SEED`              | Graine pour les modèles llama-cpp. Par défaut 0 (aléatoire). |
| `--cache-capacity CACHE_CAPACITY`    | Capacité de cache maximale. Exemples : 2000MiB, 2GiB. Lorsqu'il est fourni sans unités, les octets sont supposés. |
| `--cfg-cache`                        | llamacpp_HF : Crée un cache supplémentaire pour les sollicitations CFG négatives. |
| `--no-mmap`                          | Empêche l'utilisation de mmap. |
| `--mlock`                            | Force le système à conserver le modèle en RAM. |
| `--numa`                             | Active l'allocation de tâches NUMA pour llama.cpp. |
| `--cpu`                              | Utilise la version CPU de llama-cpp-python au lieu de la version accélérée par GPU. |

#### ctransformers

| Indicateur                      | Description |
|---------------------------------|-------------|
| `--model_type MODEL_TYPE`       | Type de modèle de modèle pré-quantifié. Actuellement gpt2, gptj, gptneox, falcon, llama, mpt, starcoder (gptbigcode), dollyv2, et replit sont supportés. |

#### AutoGPTQ

| Indicateur                               | Description |
|------------------------------------------|-------------|
| `--triton`                               | Utilise Triton. |
| `--no_inject_fused_attention`            | Désactive l'utilisation de l'attention fusionnée, ce qui utilisera moins de VRAM au coût d'une inférence plus lente. |
| `--no_inject_fused_mlp`                  | Mode Triton uniquement : désactive l'utilisation du MLP fusionné, ce qui utilisera moins de VRAM au coût d'une inférence plus lente. |
| `--no_use_cuda_fp16`                     | Cela peut rendre les modèles plus rapides sur certains systèmes. |
| `--desc_act`                             | Pour les modèles qui n'ont pas de quantize_config.json, ce paramètre est utilisé pour définir s'il faut définir desc_act ou non dans BaseQuantizeConfig. |
| `--disable_exllama`                      | Désactive le noyau ExLlama, ce qui peut améliorer la vitesse d'inférence sur certains systèmes. |



#### ExLlama

| Drapeau (Flag)           | Description |
|--------------------------|-------------|
|`--gpu-split`             | Liste séparée par des virgules de la VRAM (en GB) à utiliser par périphérique GPU pour les couches du modèle, par exemple `20,7,7` |
|`--max_seq_len MAX_SEQ_LEN`| Longueur maximale de séquence. |
|`--cfg-cache`              | ExLlama_HF : Créez un cache supplémentaire pour les prompts CFG négatifs. Nécessaire pour utiliser CFG avec ce chargeur, mais pas nécessaire pour CFG avec ExLlama de base. |

#### GPTQ-for-LLaMa

| Drapeau (Flag)                              | Description |
|---------------------------------------------|-------------|
| `--wbits WBITS`                             | Charge un modèle pré-quantifié avec une précision spécifiée en bits. 2, 3, 4 et 8 sont pris en charge. |
| `--model_type MODEL_TYPE`                   | Type de modèle pré-quantifié. Actuellement, LLaMA, OPT, et GPT-J sont pris en charge. |
| `--groupsize GROUPSIZE`                     | Taille du groupe. |
| `--pre_layer PRE_LAYER [PRE_LAYER ...]`     | Le nombre de couches à allouer au GPU. La définition de ce paramètre active l'offloading CPU pour les modèles 4 bits. Pour plusieurs GPU, écrivez les numéros séparés par des espaces, par exemple `--pre_layer 30 60`. |
| `--checkpoint CHECKPOINT`                   | Le chemin vers le fichier de point de contrôle quantifié. Si non spécifié, il sera détecté automatiquement. |
| `--monkey-patch`                            | Applique le monkey patch pour utiliser les LoRAs avec des modèles quantifiés. |

#### DeepSpeed

| Drapeau (Flag)                              | Description |
|---------------------------------------------|-------------|
| `--deepspeed`                               | Active l'utilisation de DeepSpeed ZeRO-3 pour l'inférence via l'intégration Transformers. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR`       | DeepSpeed : Répertoire à utiliser pour le déchargement ZeRO-3 NVME. |
| `--local_rank LOCAL_RANK`                   | DeepSpeed : Argument facultatif pour les configurations distribuées. |

[A propos de deepspeed](DeepSpeed.md)

#### RWKV

| Drapeau (Flag)                              | Description |
|---------------------------------------------|-------------|
| `--rwkv-strategy RWKV_STRATEGY`             | RWKV : La stratégie à utiliser lors du chargement du modèle. Exemples : "cpu fp32", "cuda fp16", "cuda fp16i8". |
| `--rwkv-cuda-on`                            | RWKV : Compile le noyau CUDA pour de meilleures performances. |

#### RoPE (pour llama.cpp, ExLlama, ExLlamaV2, et transformers)

| Drapeau (Flag)                              | Description |
|---------------------------------------------|-------------|
| `--alpha_value ALPHA_VALUE`                 | Facteur alpha des plongements positionnels pour le dimensionnement NTK RoPE. Utilisez soit cela, soit compress_pos_emb, mais pas les deux. |
| `--rope_freq_base ROPE_FREQ_BASE`           | Si supérieur à 0, sera utilisé à la place de alpha_value. Ces deux éléments sont liés par rope_freq_base = 10000 * alpha_value ^ (64 / 63). |
| `--compress_pos_emb COMPRESS_POS_EMB`       | Facteur de compression des plongements positionnels. Devrait être défini comme (longueur du contexte) / (longueur du contexte original du modèle). Égal à 1/rope_freq_scale. |

#### Gradio

| Drapeau (Flag)                              | Description |
|---------------------------------------------|-------------|
| `--listen`                                  | Rend l'interface web accessible depuis votre réseau local. |
| `--listen-host LISTEN_HOST`                 | Le nom d'hôte que le serveur utilisera. |
| `--listen-port LISTEN_PORT`                 | Le port d'écoute que le serveur utilisera. |
| `--share`                                   | Créez une URL publique. Ceci est utile pour exécuter l'interface web sur Google Colab ou similaire. |
| `--auto-launch`                             | Ouvrez l'interface web dans le navigateur par défaut au lancement. |
| `--gradio-auth USER:PWD`                    | définissez l'authentification gradio comme "username:password"; ou séparez-en plusieurs par des virgules comme "u1:p1,u2:p2,u3:p3" |
| `--gradio-auth-path GRADIO_AUTH_PATH`       | Définissez le chemin du fichier d'authentification gradio. Le fichier doit contenir un ou plusieurs ensembles utilisateur:mot de passe dans ce format : "u1:p1,u2:p2,u3:p3" |
| `--ssl-keyfile SSL_KEYFILE`                 | Le chemin du fichier de clé du certificat SSL. |
| `--ssl-certfile SSL_CERTFILE`               | Le chemin du fichier de certificat SSL. |



#### API

| Drapeau (Flag)                             | Description |
|--------------------------------------------|-------------|
| `--api`                                    | Active l'extension API. |
| `--public-api`                             | Créez une URL publique pour l'API en utilisant Cloudfare. |
| `--public-api-id PUBLIC_API_ID`            | ID du tunnel pour le Tunnel Cloudflare nommé. À utiliser avec l'option public-api. |
| `--api-blocking-port BLOCKING_PORT`        | Le port d'écoute pour l'API bloquante. |
| `--api-streaming-port STREAMING_PORT`      | Le port d'écoute pour l'API en streaming. |

#### Multimodal

| Drapeau (Flag)                             | Description |
|--------------------------------------------|-------------|
| `--multimodal-pipeline PIPELINE`           | La chaîne multimodale à utiliser. Exemples : `llava-7b`, `llava-13b`. |

## Préréglages
[Retour au sommaire](#Sommaire)

Les paramètres d'inférence peuvent être créés sous `presets/` sous forme de fichiers yaml. Ces fichiers sont détectés automatiquement au démarrage.

Les pré-réglages inclus par défaut sont le résultat d'un concours qui a reçu 7215 votes. Plus de détails peuvent être trouvés [ici](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md).

## Contribuer

Si vous souhaitez contribuer au projet, consultez les [Directives de contribution](https://github.com/oobabooga/text-generation-webui/wiki/Contributing-guidelines).

## Communauté

* Subreddit : https://www.reddit.com/r/oobabooga/
* Discord : https://discord.gg/jwZCF2dPQN

## Remerciements

En août 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) a fourni une subvention généreuse pour encourager et soutenir mon travail indépendant sur ce projet. Je suis **extrêmement** reconnaissant pour leur confiance et reconnaissance, ce qui me permettra de consacrer plus de temps à réaliser le plein potentiel de text-generation-webui.

[Retour au sommaire](#Sommaire)
