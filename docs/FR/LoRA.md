# LoRA

LoRA (Low-Rank Adaptation) est une méthode extrêmement puissante pour personnaliser un modèle de base en n'entraînant qu'un petit nombre de paramètres. Ils peuvent être attachés aux modèles lors de l'exécution.

Par exemple, un LoRA de 50 Mo peut enseigner à LLaMA une toute nouvelle langue, un style d'écriture donné, ou lui donner la capacité de suivre des instructions ou de chatter.

Voici l'état actuel de l'intégration de LoRA dans l'interface web :

| Chargeur | Statut |
|--------|------|
| Transformers | Support complet en 16 bits, `--load-in-8bit`, `--load-in-4bit`, et modes CPU. |
| ExLlama | Support d'un seul LoRA. Rapide pour retirer le LoRA ensuite. |
| AutoGPTQ | Support d'un seul LoRA. La suppression du LoRA nécessite de recharger le modèle entier.|
| GPTQ-pour-LLaMa | Support complet avec le [monkey patch](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#using-loras-with-gptq-for-llama). |

## Téléchargement d'un LoRA

Le script de téléchargement peut être utilisé. Par exemple :

```
python download-model.py tloen/alpaca-lora-7b
```

Les fichiers seront sauvegardés dans `loras/tloen_alpaca-lora-7b`.

## Utilisation du LoRA

L'option `--lora` en ligne de commande peut être utilisée. Exemples :

```
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --load-in-8bit
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --load-in-4bit
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --cpu
```

Au lieu d'utiliser l'option `--lora` en ligne de commande, vous pouvez également sélectionner le LoRA dans l'onglet "Paramètres" de l'interface.

## Invite (Prompt)

Pour le LoRA Alpaca en particulier, l'invite doit être formatée comme ceci :

```
Ci-dessous se trouve une instruction décrivant une tâche. Écrivez une réponse qui complète correctement la demande.
### Instruction:
Écrivez un script Python qui génère du texte en utilisant la bibliothèque transformers.
### Réponse:
```

Exemple de sortie :

```
Ci-dessous se trouve une instruction décrivant une tâche. Écrivez une réponse qui complète correctement la demande.
### Instruction:
Écrivez un script Python qui génère du texte en utilisant la bibliothèque transformers.
### Réponse:

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
texts = ["Bonjour le monde", "Comment ça va"]
for sentence in texts:
sentence = tokenizer(sentence)
print(f"Généré {len(sentence)} tokens à partir de '{sentence}'")
output = model(sentences=sentence).predict()
print(f"Prédit {len(output)} tokens pour '{sentence}':\n{output}")
```

## Entraînement d'un LoRA

Vous pouvez entraîner vos propres LoRAs depuis l'onglet `Entraînement`. Voir [Entraînement des LoRAs](Training-LoRAs.md) pour plus de détails.