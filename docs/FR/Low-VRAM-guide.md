Si votre GPU n'est pas assez grand pour accueillir un modèle 16 bits, essayez les solutions suivantes dans cet ordre :

### Charger le modèle en mode 8 bits

```
python server.py --load-in-8bit
```

### Charger le modèle en mode 4 bits

```
python server.py --load-in-4bit
```

### Divisez le modèle entre votre GPU et CPU

```
python server.py --auto-devices
```

Si vous pouvez charger le modèle avec cette commande mais qu'il manque de mémoire lorsque vous essayez de générer du texte, essayez de limiter progressivement la quantité de mémoire allouée au GPU jusqu'à ce que l'erreur cesse de se produire :

```
python server.py --auto-devices --gpu-memory 10
python server.py --auto-devices --gpu-memory 9
python server.py --auto-devices --gpu-memory 8
...
```

où le nombre est en GiB.

Pour un contrôle plus précis, vous pouvez également spécifier l'unité en MiB explicitement :

```
python server.py --auto-devices --gpu-memory 8722MiB
python server.py --auto-devices --gpu-memory 4725MiB
python server.py --auto-devices --gpu-memory 3500MiB
...
```

### Envoyez des couches vers un cache disque

En dernier recours, vous pouvez diviser le modèle entre votre GPU, CPU, et disque :

```
python server.py --auto-devices --disk
```

Avec cela, je suis capable de charger un modèle 30b sur mon RTX 3090, mais il faut 10 secondes pour générer 1 mot.

### DeepSpeed (expérimental)

Une alternative expérimentale à tout ce qui précède est d'utiliser DeepSpeed : [guide](DeepSpeed.md).