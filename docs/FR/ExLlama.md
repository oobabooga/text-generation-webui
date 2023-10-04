
# ExLlama

### À propos

ExLlama est un backend GPTQ extrêmement optimisé pour les modèles LLaMA. Il présente une utilisation beaucoup plus faible de la VRAM et des vitesses beaucoup plus élevées car il ne repose pas sur le code non optimisé des transformateurs.

### Utilisation

Configurez text-generation-webui pour utiliser exllama via l'UI ou la ligne de commande :
   - Dans l'onglet "Modèle", définissez "Loader" sur "exllama"
   - Spécifiez `--loader exllama` sur la ligne de commande

### Configuration manuelle

Aucune étape d'installation supplémentaire n'est nécessaire puisqu'un package exllama est déjà inclus dans le requirements.txt. Si ce package ne s'installe pas pour une raison quelconque, vous pouvez l'installer manuellement en clonant le dépôt original dans votre dossier `repositories/` :

```
mkdir repositories
cd repositories
git clone https://github.com/turboderp/exllama
```
