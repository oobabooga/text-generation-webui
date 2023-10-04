LLaMA est un modèle linguistique de grande taille développé par Meta AI.

Il a été entraîné sur plus de tokens que les modèles précédents. Le résultat est que la plus petite version avec 7 milliards de paramètres a des performances similaires à GPT-3 avec 175 milliards de paramètres.

Ce guide couvrira l'utilisation à travers l'implémentation officielle `transformers`. Pour le mode 4 bits, rendez-vous sur [Modèles GPTQ (mode 4 bits)](GPTQ-models-(4-bit-mode).md).

## Obtenir les poids

### Option 1 : poids pré-convertis

* Téléchargement direct (recommandé) :

https://huggingface.co/Neko-Institute-of-Science/LLaMA-7B-HF

https://huggingface.co/Neko-Institute-of-Science/LLaMA-13B-HF

https://huggingface.co/Neko-Institute-of-Science/LLaMA-30B-HF

https://huggingface.co/Neko-Institute-of-Science/LLaMA-65B-HF

* Torrent :

https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789

Les fichiers tokenizer dans le torrent ci-dessus sont obsolètes, en particulier les fichiers appelés `tokenizer_config.json` et `special_tokens_map.json`. Vous pouvez trouver ces fichiers ici : https://huggingface.co/oobabooga/llama-tokenizer

### Option 2 : convertissez les poids vous-même

1. Installez la bibliothèque `protobuf` :

```
pip install protobuf==3.20.1
```

2. Utilisez le script ci-dessous pour convertir le modèle au format `.pth` que vous, cher collègue universitaire, avez téléchargé à l'aide du lien officiel de Meta.

Si vous avez `transformers` installé :

```
python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir /chemin/vers/LLaMA --model_size 7B --output_dir /tmp/outputs/llama-7b
```

Sinon, téléchargez d'abord [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) et exécutez :

```
python convert_llama_weights_to_hf.py --input_dir /chemin/vers/LLaMA --model_size 7B --output_dir /tmp/outputs/llama-7b
```

3. Déplacez le dossier `llama-7b` dans votre dossier `text-generation-webui/models`.

## Démarrer l'interface web

```python
python server.py --model llama-7b
```