> **RWKV : RNN avec des performances LLM au niveau du Transformer**
>
> Il combine le meilleur du RNN et du transformateur - d'excellentes performances, une inférence rapide, économise la VRAM, une formation rapide, une ctx_len "infinie", et un embedding de phrase gratuit (en utilisant l'état caché final).

https://github.com/BlinkDL/RWKV-LM

https://github.com/BlinkDL/ChatRWKV

## Utilisation de RWKV dans l'interface web

### Poids de Hugging Face

Il suffit de télécharger les poids depuis https://huggingface.co/RWKV et de les charger comme vous le feriez pour n'importe quel autre modèle.

Il y a un bug dans transformers==4.29.2 qui empêche de charger RWKV en mode 8 bits. Vous pouvez installer la branche de développement pour résoudre ce bug : `pip install git+https://github.com/huggingface/transformers`.

### Poids .pth originaux

Les instructions ci-dessous datent d'avant la prise en charge de RWKV dans les transformers et sont conservées à des fins historiques. L'ancienne implémentation est peut-être plus rapide, mais elle n'offre pas la gamme complète de sampleurs que propose la bibliothèque transformers.

#### 0. Installez la bibliothèque RWKV

```
pip install rwkv
```

`0.7.3` était la dernière version que j'ai testée. Si vous rencontrez des problèmes, essayez ```pip install rwkv==0.7.3```.

#### 1. Téléchargez le modèle

Il est disponible en différentes tailles :

* https://huggingface.co/BlinkDL/rwkv-4-pile-3b/
* https://huggingface.co/BlinkDL/rwkv-4-pile-7b/
* https://huggingface.co/BlinkDL/rwkv-4-pile-14b/

Il existe également d'anciennes versions de tailles plus petites comme :

* https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth

Téléchargez le `.pth` choisi et placez-le directement dans le dossier `models`.

#### 2. Téléchargez le tokenizer

[20B_tokenizer.json](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/20B_tokenizer.json)

Placez-le également directement dans le dossier `models`. Assurez-vous de ne pas le renommer. Il doit être appelé `20B_tokenizer.json`.

#### 3. Lancez l'interface web

Aucune étape supplémentaire n'est requise. Lancez-la comme vous le feriez avec n'importe quel autre modèle.

```
python server.py --listen --no-stream --model RWKV-4-Pile-169M-20220807-8023.pth
```

#### Définir une stratégie personnalisée

Il est possible d'avoir un contrôle très précis sur l'externalisation et la précision du modèle avec le drapeau `--rwkv-strategy`. Les valeurs possibles incluent :

```
"cpu fp32" # Mode CPU
"cuda fp16" # Mode GPU avec précision float16
"cuda fp16 *30 -> cpu fp32" # Externalisation GPU+CPU. Plus le nombre après * est élevé, plus l'allocation GPU est importante.
"cuda fp16i8" # Mode GPU avec précision 8 bits
```

Consultez le README du package PyPl pour plus de détails : https://pypi.org/project/rwkv/

#### Compilation du noyau CUDA

Vous pouvez compiler le noyau CUDA pour le modèle avec `--rwkv-cuda-on`. Cela devrait beaucoup améliorer les performances, mais je n'ai pas encore réussi à le faire fonctionner.
