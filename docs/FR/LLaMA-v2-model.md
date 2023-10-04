# LLaMA-v2

Pour convertir LLaMA-v2 du format `.pth` fourni par Meta au format des transformateurs, suivez les étapes ci-dessous :

1) Utilisez la commande `cd` pour vous déplacer dans votre dossier `llama` (celui contenant `download.sh` et les modèles que vous avez téléchargés) :

```
cd llama
```

2) Clonez la bibliothèque des transformateurs :

```
git clone 'https://github.com/huggingface/transformers'
```

3) Créez des liens symboliques des dossiers téléchargés vers des noms que le script de conversion peut reconnaître :

```
ln -s llama-2-7b 7B
ln -s llama-2-13b 13B
```

4) Effectuez les conversions :

```
mkdir llama-2-7b-hf llama-2-13b-hf
python ./transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir . --model_size 7B --output_dir llama-2-7b-hf --safe_serialization true
python ./transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir . --model_size 13B --output_dir llama-2-13b-hf --safe_serialization true
```

5) Déplacez les dossiers de sortie à l'intérieur de `text-generation-webui/models`

6) Amusez-vous !