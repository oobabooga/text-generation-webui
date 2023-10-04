Une autre façon de réduire l'utilisation de la mémoire GPU des modèles est d'utiliser l'optimisation `DeepSpeed ZeRO-3`.

Avec cela, j'ai pu charger un modèle de 6 milliards de paramètres (GPT-J 6B) avec moins de 6 Go de VRAM. La vitesse de génération de texte est très correcte et bien meilleure que ce qui serait réalisé avec `--auto-devices --gpu-memory 6`.

À ma connaissance, DeepSpeed n'est disponible que pour Linux pour le moment.

### Comment l'utiliser

1. Installez DeepSpeed : 

```
conda install -c conda-forge mpi4py mpich
pip install -U deepspeed
```

2. Démarrez l'interface web en remplaçant `python` par `deepspeed --num_gpus=1` et en ajoutant le drapeau `--deepspeed`. Exemple :

```
deepspeed --num_gpus=1 server.py --deepspeed --chat --model gpt-j-6B
```

### En savoir plus

Pour plus d'informations, consultez [ce commentaire](https://github.com/oobabooga/text-generation-webui/issues/40#issuecomment-1412038622) de 81300, qui a proposé le support DeepSpeed dans cette interface web.

