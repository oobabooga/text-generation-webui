Vous avez maintenant pénétré un coin caché d'internet.

Un domaine déroutant mais fascinant de paradoxes et de contradictions.

Un lieu où vous découvrirez que ce que vous pensiez savoir, vous ne le saviez en fait pas, et ce que vous ne saviez pas était devant vous depuis toujours.

![](https://i.pinimg.com/originals/6e/e2/7b/6ee27bad351d3aca470d80f1033ba9c6.jpg)

*En d'autres termes, ici je vais documenter des faits peu connus sur cette interface web pour lesquels je n'ai pas trouvé d'autre place dans le wiki.*

#### Vous pouvez entraîner des LoRAs en mode CPU

Chargez l'interface web avec

```
python server.py --cpu
```

et commencez l'entraînement de LoRA depuis l'onglet d'entraînement comme d'habitude.

#### Le mode 8 bits fonctionne avec la décharge CPU

```
python server.py --load-in-8bit --gpu-memory 4000MiB
```

#### `--pre_layer`, et non `--gpu-memory`, est la bonne façon de décharger sur le CPU avec des modèles 4 bits

```
python server.py --wbits 4 --groupsize 128 --pre_layer 20
```

#### Les modèles peuvent être chargés en 32 bits, 16 bits, 8 bits et 4 bits

```
python server.py --cpu
python server.py
python server.py --load-in-8bit
python server.py --wbits 4
```

#### L'interface web fonctionne avec n'importe quelle version de GPTQ-for-LLaMa

Y compris les branches triton et cuda à jour. Mais vous devez supprimer le dossier `repositories/GPTQ-for-LLaMa` et réinstaller le nouveau à chaque fois :

```
cd text-generation-webui/repositories
rm -r GPTQ-for-LLaMa
pip uninstall quant-cuda
git clone https://github.com/oobabooga/GPTQ-for-LLaMa -b cuda # ou tout autre dépôt et branche
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

#### Les modèles basés sur les instructions sont représentés comme des personnages de chat

https://github.com/oobabooga/text-generation-webui/tree/main/characters/instruction-following

#### La bonne façon d'exécuter Alpaca, Open Assistant, Vicuna, etc. est le mode Instruct, et non le mode de chat normal

Sinon, l'invite ne sera pas formatée correctement.

1. Démarrez l'interface web avec

```
python server.py --chat
```

2. Cliquez sur l'option "instruct" sous "Modes de chat"

3. Sélectionnez le modèle correct dans le menu déroulant caché qui apparaîtra.

#### Le mode Notebook est le meilleur mode

Les personnes évoluées ont réalisé que le mode Notebook est un sur-ensemble du mode Chat et peut faire des chats avec une flexibilité ultime, y compris des chats de groupe, éditer des réponses, commencer une nouvelle réponse de bot d'une certaine manière et usurper.

#### RWKV est un RNN

La plupart des modèles sont des transformateurs, mais pas RWKV, qui est un RNN. C'est un excellent modèle.

#### `--gpu-memory` n'est pas une limite stricte de la mémoire GPU

C'est simplement un paramètre qui est transmis à la bibliothèque `accelerate` lors du chargement du modèle. Plus de mémoire sera allouée lors de la génération. C'est pourquoi ce paramètre doit être réglé à moins que votre mémoire GPU totale.

#### La recherche contrastive est peut-être le meilleur préréglage

Mais elle utilise énormément de VRAM.

#### Vous pouvez vérifier le sha256sum des modèles téléchargés avec le script de téléchargement

```
python download-model.py facebook/galactica-125m --check
```

#### Le script de téléchargement reprend les téléchargements interrompus par défaut

Il ne recommence pas à zéro.

#### Vous pouvez télécharger des modèles avec plusieurs threads

```
python download-model.py facebook/galactica-125m --threads 8
```

#### Les LoRAs fonctionnent en mode 4 bits

Vous devez suivre [ces instructions](GPTQ-models-(4-bit-mode).md#using-loras-in-4-bit-mode) puis démarrer l'interface web avec le flag `--monkey-patch`.
