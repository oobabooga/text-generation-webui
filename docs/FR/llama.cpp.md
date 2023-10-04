# llama.cpp

llama.cpp est le meilleur backend dans deux scénarios importants :

1) Vous n'avez pas de GPU.
2) Vous souhaitez exécuter un modèle qui ne rentre pas dans votre GPU.

## Configuration des modèles

#### Pré-converti

Téléchargez les modèles GGUF directement dans votre dossier `text-generation-webui/models`. Ce sera un fichier unique.

* Assurez-vous que son nom se termine par `.gguf`.
* La quantification `q4_K_M` est recommandée.

#### Convertissez Llama vous-même

Suivez les instructions dans le README de llama.cpp pour générer un GGUF : https://github.com/ggerganov/llama.cpp#prepare-data--run

## Accélération GPU

Activée avec le paramètre `--n-gpu-layers`.

* Si vous avez suffisamment de VRAM, utilisez un nombre élevé comme `--n-gpu-layers 1000` pour transférer toutes les couches vers le GPU.
* Sinon, commencez avec un nombre faible comme `--n-gpu-layers 10` puis augmentez-le progressivement jusqu'à épuisement de la mémoire.

Cette fonctionnalité fonctionne immédiatement pour les GPU NVIDIA sur Linux (amd64) ou Windows. Pour les autres GPU, vous devez désinstaller `llama-cpp-python` avec

```
pip uninstall -y llama-cpp-python
```

puis le recompiler en utilisant les commandes ici : https://pypi.org/project/llama-cpp-python/

#### macOS

Pour macOS, voici les commandes :

```
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```
