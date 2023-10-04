
# Extensions

Les extensions sont définies par des fichiers nommés `script.py` situés dans des sous-dossiers de `text-generation-webui/extensions`. Elles sont chargées au démarrage si le nom du dossier est spécifié après le drapeau `--extensions`.

Par exemple, `extensions/silero_tts/script.py` est chargé avec `python server.py --extensions silero_tts`.

## [text-generation-webui-extensions](https://github.com/oobabooga/text-generation-webui-extensions)

Le dépôt ci-dessus contient un annuaire d'extensions utilisateurs.

Si vous créez une extension, vous êtes invité à l'héberger dans un dépôt GitHub et à soumettre une PR pour l'ajouter à la liste.

## Extensions intégrées

|Extension|Description|
|---------|-----------|
|[api](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/api)| Crée une API avec deux points d'accès, l'un pour le streaming à `/api/v1/stream` port 5005 et l'autre pour le blocage à `/api/v1/generate` port 5000. C'est l'API principale pour le webui. |
|[openai](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)| Crée une API qui imite l'API OpenAI et peut être utilisée comme remplacement direct. |
|[multimodal](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal) | Ajoute le support multimodal (texte+images). Pour une description détaillée, voir [README.md](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal/README.md) dans le répertoire d'extension. |
|[google_translate](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/google_translate)| Traduit automatiquement les entrées et les sorties en utilisant Google Translate.|
|[silero_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/silero_tts)| Extension de synthèse vocale utilisant [Silero](https://github.com/snakers4/silero-models). En mode chat, les réponses sont remplacées par un widget audio. |
|[elevenlabs_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/elevenlabs_tts)| Extension de synthèse vocale utilisant l'API [ElevenLabs](https://beta.elevenlabs.io/). Vous avez besoin d'une clé API pour l'utiliser. |
|[whisper_stt](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/whisper_stt)| Vous permet d'entrer vos entrées en mode chat en utilisant votre microphone. |
|[sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures)| Vous permet de demander des images au bot en mode chat, qui seront générées en utilisant l'API AUTOMATIC1111 Stable Diffusion. Voir les exemples [ici](https://github.com/oobabooga/text-generation-webui/pull/309). |
|[character_bias](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/character_bias)| Juste un exemple très simple qui ajoute une chaîne cachée au début de la réponse du bot en mode chat. |
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/)| Crée un champ de téléchargement d'image qui peut être utilisé pour envoyer des images au bot en mode chat. Les légendes sont automatiquement générées à l'aide de BLIP. |
|[gallery](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/gallery/)| Crée une galerie avec les personnages du chat et leurs images. |
|[superbooga](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superbooga)| Une extension qui utilise ChromaDB pour créer un pseudocontexte arbitrairement grand, en prenant comme entrée des fichiers texte, des URL ou du texte collé. Basé sur https://github.com/kaiokendev/superbig. |
|[ngrok](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/ngrok)| Vous permet d'accéder à l'interface utilisateur web à distance en utilisant le service de tunnel inversé ngrok (gratuit). C'est une alternative à la fonction `--share` intégrée à Gradio. |
|[perplexity_colors](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/perplexity_colors)| Colore chaque token dans le texte de sortie en fonction de sa probabilité associée, telle qu'elle est dérivée des logits du modèle. |

## Comment écrire une extension


La grille des extensions est basé sur des fonctions et des variables spéciales que vous pouvez définir dans `script.py`. Les fonctions sont les suivantes:

| Fonction        | Description |
|-------------|-------------|
| `def setup()` | Est exécuté lorsque l'extension est importée. |
| `def ui()` | Crée des éléments gradio personnalisés lorsque l'UI est lancée. | 
| `def custom_css()` | Renvoie du CSS personnalisé sous forme de chaîne. Il est appliqué chaque fois que l'UI web est chargée. |
| `def custom_js()` | Idem que ci-dessus, mais pour le javascript. |
| `def input_modifier(string, state, is_chat=False)`  | Modifie la chaîne d'entrée avant qu'elle n'entre dans le modèle. En mode chat, il est appliqué au message de l'utilisateur. Sinon, il est appliqué à l'ensemble de l'invite. |
| `def output_modifier(string, state, is_chat=False)`  | Modifie la chaîne de sortie avant qu'elle ne soit présentée dans l'UI. En mode chat, il est appliqué à la réponse du bot. Sinon, il est appliqué à la totalité de la sortie. |
| `def chat_input_modifier(text, visible_text, state)` | Modifie à la fois les entrées visibles et internes en mode chat. Peut être utilisé pour détourner l'entrée du chat avec un contenu personnalisé. |
| `def bot_prefix_modifier(string, state)`  | Appliqué en mode chat au préfixe de la réponse du bot. |
| `def state_modifier(state)`  | Modifie le dictionnaire contenant les paramètres d'entrée de l'UI avant qu'il ne soit utilisé par les fonctions de génération de texte. |
| `def history_modifier(history)`  | Modifie l'historique du chat avant que la génération de texte en mode chat ne commence. |
| `def custom_generate_reply(...)` | Remplace la fonction principale de génération de texte. |
| `def custom_generate_chat_prompt(...)` | Remplace le générateur d'invite en mode chat. |
| `def tokenizer_modifier(state, prompt, input_ids, input_embeds)` | Modifie les `input_ids`/`input_embeds` alimentés au modèle. Doit renvoyer `prompt`, `input_ids`, `input_embeds`. Voir l'extension `multimodal` pour un exemple. |
| `def custom_tokenized_length(prompt)` | Utilisé en conjonction avec `tokenizer_modifier`, renvoie la longueur en tokens de `prompt`. Voir l'extension `multimodal` pour un exemple. |

De plus, vous pouvez définir un dictionnaire `params` spécial. Dans celui-ci, la clé `display_name` est utilisée pour définir le nom affiché de l'extension dans l'UI, et la clé `is_tab` est utilisée pour définir si l'extension doit apparaître dans un nouvel onglet. Par défaut, les extensions apparaissent au bas de l'onglet "Génération de texte".

Exemple :

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
}
```

Le dictionnaire `params` peut également contenir des variables que vous souhaitez personnaliser via un fichier `settings.yaml`. Par exemple, en supposant que l'extension se trouve dans `extensions/google_translate`, la variable `language string` dans

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
    "language string": "jp"
}
```

peut être personnalisée en ajoutant une clé appelée `google_translate-language string` à `settings.yaml`:

```python
google_translate-language string: 'fr'
``` 

C'est-à-dire que la syntaxe pour la clé est `nom_extension-nom_variable`.

## Utilisation de plusieurs extensions en même temps

Vous pouvez activer plusieurs extensions à la fois en fournissant leurs noms séparés par des espaces après `--extensions`. Les modificateurs d'entrée, de sortie et de préfixe de bot seront appliqués dans l'ordre spécifié. 

Exemple :

```
python server.py --extensions enthusiasm translate # Appliquer d'abord enthusiasm, puis translate
python server.py --extensions translate enthusiasm # Appliquer d'abord translate, puis enthusiasm
```

Notez bien que pour :
- `custom_generate_chat_prompt`
- `custom_generate_reply`
- `custom_tokenized_length`

seule la première déclaration rencontrée sera utilisée et le reste sera ignoré.


## Un exemple complet

Le code source ci-dessous peut être trouvé à l'adresse [extensions/example/script.py](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/example/script.py).
Voici le code traduit avec les commentaires traduits :

```python
"""
Un exemple d'extension. Elle ne fait rien, mais vous pouvez ajouter des transformations
avant les déclarations de retour pour personnaliser le comportement du webui.

À partir de history_modifier et se terminant à output_modifier, les
fonctions sont déclarées dans le même ordre qu'elles sont appelées au
moment de la génération.
"""

import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "Extension Exemple",
    "is_tab": False,
}

class MyLogits(LogitsProcessor):
    """
    Manipule les probabilités pour le prochain token avant qu'il ne soit échantillonné.
    Utilisé dans la fonction logits_processor_modifier ci-dessous.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifie l'historique du chat.
    Utilisé uniquement en mode chat.
    """
    return history

def state_modifier(state):
    """
    Modifie la variable d'état, qui est un dictionnaire contenant les valeurs d'entrée
    dans l'UI comme les curseurs et les cases à cocher.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifie la chaîne d'entrée de l'utilisateur en mode chat (visible_text).
    Vous pouvez également modifier la représentation interne de l'utilisateur
    entrée (texte) pour changer son apparence dans l'invite.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    Dans les modes par défaut/notebook, modifie l'ensemble de l'invite.

    En mode chat, c'est la même chose que chat_input_modifier mais seulement appliqué
    à "texte", ici appelé "chaîne", et non à "visible_text".
    """
    return string

def bot_prefix_modifier(string, state):
    """
    Modifie le préfixe pour la prochaine réponse du bot en mode chat.
    Par défaut, le préfixe sera quelque chose comme "Nom du Bot:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifie les ids d'entrée et les incrustations.
    Utilisé par l'extension multimodale pour mettre les incrustations d'image dans l'invite.
    Utilisé uniquement par les chargeurs qui utilisent la bibliothèque transformers pour l'échantillonnage.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Ajoute des processeurs de logits à la liste, vous permettant d'accéder et de modifier
    les probabilités du prochain token.
    Utilisé uniquement par les chargeurs qui utilisent la bibliothèque transformers pour l'échantillonnage.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifie la sortie LLM avant qu'elle ne soit présentée.

    En mode chat, la version modifiée va dans history['visible'],
    et la version originale va dans history['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Remplace la fonction qui génère l'invite à partir de l'historique du chat.
    Utilisé uniquement en mode chat.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Retourne une chaîne CSS qui est ajoutée à la CSS pour le webui.
    """
    return ''

def custom_js():
    """
    Retourne une chaîne javascript qui est ajoutée au javascript
    pour le webui.
    """
    return ''

def setup():
    """
    Est exécuté une seule fois, lorsque l'extension est importée.
    """
    pass

def ui():
    """
    Est exécuté lorsque l'UI est dessinée. Les éléments gradio personnalisés et
    leurs gestionnaires d'événements correspondants doivent être définis ici.

    Pour en savoir plus sur les composants gradio, consultez la documentation :
    https://gradio.app/docs/
    """
    pass
```
