L'interface web vise à faciliter autant que possible la formation de vos propres LoRAs. Tout se résume à quelques étapes simples :

### **Étape 1** : Établir un plan.
- Quel modèle de base souhaitez-vous utiliser ? Le LoRA que vous créez doit être associé à une seule architecture (par exemple LLaMA-13B) et ne peut être transféré à d'autres (par exemple LLaMA-7B, StableLM, etc. seraient tous différents). Les dérivés du même modèle (par exemple le peaufinage Alpaca de LLaMA-13B) pourraient être transférables, mais même dans ce cas, il est préférable de s'entraîner exactement sur ce que vous prévoyez d'utiliser.
- Quel format de modèle souhaitez-vous ? Au moment de la rédaction, les modèles 8 bits sont les plus stables, et les 4 bits sont pris en charge mais sont expérimentaux. Dans un futur proche, il est probable que 4 bits seront la meilleure option pour la plupart des utilisateurs.
- Sur quoi le formez-vous ? Voulez-vous qu'il apprenne des informations réelles, un format simple, ...?

### **Étape 2** : Rassemblez un ensemble de données.
- Si vous utilisez un ensemble de données similaire au format [Alpaca](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json), il est pris en charge nativement par l'entrée `Formatted Dataset` de l'interface web, avec des options de formatage prédéfinies.
- Si vous utilisez un ensemble de données qui ne correspond pas au format d'Alpaca, mais qui utilise la même structure JSON de base, vous pouvez créer votre propre fichier de format en copiant `training/formats/alpaca-format.json` dans un nouveau fichier et en [modifiant son contenu](#format-files).
- Si vous pouvez obtenir l'ensemble de données dans un simple fichier texte, cela fonctionne aussi ! Vous pouvez vous former en utilisant l'option d'entrée `Raw text file`.
    - Cela signifie que vous pouvez par exemple simplement copier/coller un chatlog/page de documentation/ce que vous voulez, le mettre dans un fichier texte brut, et vous former dessus.
- Si vous utilisez un ensemble de données structurées non dans ce format, vous devrez peut-être trouver un moyen externe de le convertir - ou ouvrir un problème pour demander un support natif.

### **Étape 3** : Effectuez l'entrainement.
- **3.1** : Chargez l'interface web, et votre modèle.
    - Assurez-vous de ne pas avoir déjà de LoRAs chargés (sauf si vous souhaitez train en multi-LoRA).
- **3.2** : Ouvrez l'onglet `training` en haut, sous-onglet `Train LoRA`.
- **3.3** : Remplissez le nom du LoRA, sélectionnez votre ensemble de données dans les options d'ensemble de données.
- **3.4** : Sélectionnez d'autres paramètres selon vos préférences. Voir [paramètres ci-dessous](#parameters).
- **3.5** : cliquez sur `Start LoRA Training`, et attendez.
    - Cela peut prendre quelques heures pour un grand ensemble de données, ou seulement quelques minutes pour une petite exécution.
    - Vous voudrez peut-être surveiller votre [value loss](#loss) pendant ce temps.

### **Étape 4** : Évaluez vos résultats.
- Chargez le LoRA sous l'onglet Modèles.
- Vous pouvez le tester sur l'onglet `text generation`, ou utiliser le sous-onglet `Perplexity evaluation` de l'onglet `training`.
- Si vous avez utilisé l'option `Save every n steps`, vous pouvez récupérer des copies antérieures du modèle dans les sous-dossiers du dossier du modèle LoRA et les essayer à la place.

### **Étape 5** : Relancez si vous n'êtes pas satisfait.
- Assurez-vous de décharger le LoRA avant de le former.
- Vous pouvez simplement reprendre une exécution précédente - utilisez `copy settings` pour sélectionner votre LoRA, et modifiez les paramètres. Notez que vous ne pouvez pas changer le `rank` d'un LoRA déjà créé.
    - Si vous souhaitez reprendre à partir d'un point de contrôle enregistré en cours de route, copiez simplement le contenu du dossier de point de contrôle dans le dossier du LoRA.
    - (Remarque : `adapter_model.bin` est le fichier important qui contient le contenu réel du LoRA).
    - Cela réinitialisera le taux d'apprentissage et les étapes au début. Si vous

 souhaitez reprendre comme si vous étiez en cours de route, vous pouvez ajuster votre taux d'apprentissage au dernier LR signalé dans les journaux et réduire vos époques.
- Ou, vous pouvez recommencer entièrement si vous préférez.
- Si votre modèle produit des sorties corrompues, vous devrez probablement recommencer et utiliser un taux d'apprentissage plus bas.
- Si votre modèle n'apprend pas d'informations détaillées mais que vous le souhaitez, vous devrez peut-être simplement exécuter plus d'époques, ou vous pourriez avoir besoin d'un rang plus élevé.
- Si votre modèle impose un format que vous ne vouliez pas, vous devrez peut-être ajuster votre ensemble de données, ou recommencer et ne pas vous entraîner aussi loin.

## Fichiers de Format

Si vous utilisez des ensembles de données au format JSON, on suppose qu'ils sont dans le format approximatif suivant :

```json
[
    {
        "cle": "valeur",
        "cle2": "valeur2"
    },
    {
        // etc
    }
]
```

Où les clés (par exemple `cle`, `cle2` ci-dessus) sont standardisées et relativement cohérentes dans l'ensemble de données, et les valeurs (par exemple `valeur`, `valeur2`) contiennent le contenu réellement destiné à être formé.

Pour Alpaca, les clés sont `instruction`, `input`, et `output`, où `input` est parfois vide.

Un simple fichier de format pour Alpaca à utiliser comme chatbot est :

```json
{
    "instruction,output": "Utilisateur: %instruction%\nAssistant: %output%",
    "instruction,input,output": "Utilisateur: %instruction%: %input%\nAssistant: %output%"
}
```

Notez que les clés (par exemple `instruction,output`) sont une liste de clés de l'ensemble de données séparées par des virgules, et les valeurs sont une simple chaîne qui utilise ces clés avec `%%`.

Par exemple, si un ensemble de données a `"instruction": "répondez à ma question"`, alors le fichier de format `Utilisateur: %instruction%\n` sera automatiquement rempli par `Utilisateur: répondez à ma question\n`.

Si vous avez différents ensembles de clés d'entrée, vous pouvez créer votre propre fichier de format pour y correspondre. Ce fichier de format est conçu pour être aussi simple que possible afin de faciliter la modification selon vos besoins.

## Paramètres du Fichier Texte Brut

Lorsque vous utilisez des fichiers texte brut comme ensemble de données, le texte est automatiquement divisé en morceaux en fonction de votre `Cutoff Length` vous obtenez quelques options de base pour les configurer.
- `Overlap Length` indique combien de morceaux doivent se chevaucher. Le chevauchement des morceaux aide à éviter que le modèle apprenne d'étranges coupures en milieu de phrase, et apprend plutôt des phrases continues qui découlent du texte précédent.
- `Prefer Newline Cut Length` définit une distance maximale en caractères pour décaler la coupure de morceau vers les nouvelles lignes. Ceci aide à éviter que les lignes ne commencent ou ne se terminent en milieu de phrase, empêchant le modèle d'apprendre à couper les phrases de manière aléatoire.
- `Hard Cut String` définit une chaîne qui indique qu'il doit y avoir une coupure dure sans chevauchement. Par défaut, cela correspond à `\n\n\n`, soit 3 sauts de ligne. Aucun morceau formé ne contiendra jamais cette chaîne. Cela vous permet d'insérer des sections de texte non liées dans le même fichier texte, mais de garantir que le modèle ne sera pas formé pour changer de sujet de manière aléatoire.

## Paramètres

L'objectif et la fonction de chaque paramètre sont documentés sur la page de l'interface web, alors lisez-les dans l'UI pour comprendre vos options.

Cela dit, voici un guide des choix de paramètres les plus importants que vous devriez considérer :

### VRAM (Mémoire Vidéo)

- Tout d'abord, vous devez prendre en compte la disponibilité de votre VRAM (mémoire vidéo).
    - Généralement, avec les paramètres par défaut, l'utilisation de la VRAM pour l'entraînement avec les paramètres par défaut est très proche de celle lors de la génération de texte (avec 1000+ jetons de contexte). Si vous pouvez générer du texte, vous pouvez entraîner des LoRAs.
        - Remarque : c'est pire par défaut avec le monkeypatch 4-bit actuellement. Réduisez la `Micro Batch Size` à `1` pour revenir à ce que vous attendez.
    - Si vous avez de la VRAM en réserve, définir des tailles de lots plus élevées utilisera plus de VRAM et vous offrira un entraînement de meilleure qualité en retour.
    - Si vous avez de grandes données, définir une longueur de coupure plus élevée peut être bénéfique, mais coûtera beaucoup de VRAM. Si vous pouvez vous le permettre, réglez la taille de votre lot sur `1` et voyez jusqu'où vous pouvez pousser votre longueur de coupure.
    - Si vous manquez de VRAM, réduire la taille du lot ou la longueur de coupure améliorera bien sûr cela.
    - N'ayez pas peur d'essayer et de voir ce qui se passe. Si c'est trop, une erreur se produira, et vous pourrez baisser les paramètres et réessayer.

### Rang

- Deuxièmement, vous devez prendre en compte la quantité d'apprentissage que vous souhaitez.
    - Par exemple, vous souhaiterez peut-être simplement apprendre un format de dialogue (comme dans le cas d'Alpaca), dans ce cas, définir une faible valeur de `Rang` (32 ou moins) fonctionne très bien.
    - Ou bien, vous pourriez être en train d'entraîner de la documentation de projet que vous souhaitez que le bot comprenne et soit capable de poser des questions, auquel cas plus le rang est élevé, mieux c'est.
    - Généralement, Rang plus élevé = apprentissage plus précis = plus de contenu total appris = utilisation accrue de la VRAM pendant l'entraînement.

### Taux d'Apprentissage et Époques

- Troisièmement, la précision avec laquelle vous voulez que cela soit appris.
    - En d'autres termes, êtes-vous d'accord ou non avec le fait que le modèle perde des compréhensions non liées.
    - Vous pouvez contrôler cela avec 3 paramètres clés : le taux d'apprentissage, son planificateur, et vos époques totales.
    - Le taux d'apprentissage contrôle combien de changements sont apportés au modèle par chaque jeton qu'il voit.
        - Il est généralement exprimé en notation scientifique, donc par exemple `3e-4` signifie `3 * 10^-4` soit `0.0003`. Le nombre après `e-` contrôle combien de `0` sont dans le nombre.
        - Des valeurs plus élevées permettent à l'entraînement de s'exécuter plus rapidement, mais sont également plus susceptibles de corrompre les données précédentes dans le modèle.
    - Vous avez essentiellement deux variables à équilibrer : le LR (taux d'apprentissage) et les Époques.
        - Si vous augmentez le LR, vous pouvez réduire proportionnellement les Époques. Un LR élevé + peu d'époques = entraînement très rapide, mais de faible qualité.
        - Si vous réduisez le LR, augmentez les époques. Un faible LR + de nombreuses époques = entraînement lent mais de haute qualité.
    - Le planificateur contrôle le changement au fil du temps pendant que vous vous entraînez - il commence haut, puis descend. Cela aide à équilibrer l'ajout de données et la qualité, en même temps.
        - Vous pouvez voir des graphiques des différentes options de planificateur [dans la documentation de HuggingFace ici](https://moon-ci-docs.huggingface.co/docs/transformers/pr_1/en/main_classes/optimizer_schedules#transformers.SchedulerType).

## Loss

Lorsque vous effectuez un entraînement, la fenêtre de la console WebUI affichera des rapports qui incluent, entre autres, une valeur numérique appelée `Loss`. Elle commencera par un nombre élevé et diminuera progressivement.

La "Loss" dans le monde de la formation en IA signifie théoriquement "à quel point le modèle est proche de la perfection", où `0` signifie "absolument parfait". Cela est calculé en mesurant la différence entre la sortie exacte du modèle que vous l'entraînez à produire et ce qu'il produit réellement.

En pratique, un bon LLM devrait avoir une très complexe gamme variable d'idées fonctionnant dans sa tête artificielle, donc une perte de `0` indiquerait que le modèle s'est brisé et a oublié comment penser à autre chose que ce sur quoi vous l'avez formé.

Ainsi, en effet, la perte est un jeu d'équilibre : vous voulez qu'elle soit suffisamment basse pour qu'elle comprenne vos données, mais suffisamment élevée pour qu'elle n'oublie pas tout le reste. Généralement, si elle passe en dessous de `1.0`, elle commencera à oublier ses souvenirs précédents, et vous devriez arrêter l'entraînement. Dans certains cas, vous préférerez peut-être la descendre jusqu'à `0.5` (si vous voulez qu'elle soit très très prévisible). Différents objectifs ont des besoins différents, alors n'hésitez pas à expérimenter pour voir ce qui fonctionne le mieux pour vous.

Note : si vous voyez la perte commencer à ou soudainement sauter exactement à `0`, il est probable que quelque chose a mal tourné dans votre processus d'entraînement (par exemple, corruption du modèle).

## Note : Monkeypatch 4-Bit

Le [monkeypatch 4-bit LoRA](GPTQ-models-(4-bit-mode).md#using-loras-in-4-bit-mode) fonctionne pour l'entraînement, mais a des effets secondaires :
- L'utilisation de la VRAM est actuellement plus é

levée. Vous pouvez réduire la `Micro Batch Size` à `1` pour compenser.
- Les modèles font des choses étranges. Les LoRAs s'appliquent d'elles-mêmes, ou refusent de s'appliquer, ou génèrent spontanément des erreurs, etc. Il peut être utile de recharger le modèle de base ou de redémarrer le WebUI entre l'entraînement/l'utilisation pour minimiser les chances que quelque chose tourne mal.
- Charger ou travailler avec plusieurs LoRAs en même temps ne fonctionne pas actuellement.
- En général, considérez et traitez le monkeypatch comme le bricolage temporaire qu'il est - il fonctionne, mais n'est pas très stable. Il s'améliorera avec le temps lorsque tout sera fusionné en amont pour un soutien officiel complet.

## Notes original

La formation LoRA a été contribuée par [mcmonkey4eva](https://github.com/mcmonkey4eva) dans le PR [#570](https://github.com/oobabooga/text-generation-webui/pull/570).

### Utilisation du code alpaca-lora original

Conservé ici pour référence. L'onglet Entraînement offre beaucoup plus de fonctionnalités que cette méthode.

```
conda activate textgen
git clone https://github.com/tloen/alpaca-lora
```

Modifiez ces deux lignes dans `alpaca-lora/finetune.py` pour utiliser votre dossier de modèle existant plutôt que de télécharger tout depuis decapoda :

```
model = LlamaForCausalLM.from_pretrained(
    "models/llama-7b",
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(
    "models/llama-7b", add_eos_token=True
)
```

Exécutez le script avec :

```
python finetune.py
```

Cela fonctionne simplement. Il s'exécute à 22.32s/it, avec 1170 itérations au total, soit environ 7 heures et demie pour entraîner une LoRA. RTX 3090, 18153MiB de VRAM utilisée, consommation maximale d'énergie (350W, mode chauffage de pièce).