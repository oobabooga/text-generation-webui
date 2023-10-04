# Paramètres de Génération

Pour une description technique des paramètres, la [documentation des transformateurs](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) est une bonne référence.

Les meilleurs préréglages, selon l'expérience de la [Preset Arena](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md), sont :

**Suivi des instructions :**

1) Divine Intellect
2) Big O
3) simple-1
4) Extraterrestre
5) StarChat
6) Titanic
7) tfs-with-top-a
8) Asterism
9) Recherche Contrastée

**Chat :**

1) Énigme de Minuit
2) Yara
3) Onde Courte

### Température

Facteur principal pour contrôler le caractère aléatoire des sorties. 0 = déterministe (seul le jeton le plus probable est utilisé). Valeur plus élevée = plus d'aléatoire.

### top_p

Si ce n'est pas fixé à 1, sélectionnez des jetons dont les probabilités s'additionnent à moins que ce nombre. Valeur plus élevée = plus grande plage de résultats aléatoires possibles.

### top_k

Similaire à top_p, mais sélectionnez plutôt uniquement les jetons top_k les plus probables. Valeur plus élevée = plus grande plage de résultats aléatoires possibles.

### typical_p

Si ce n'est pas fixé à 1, ne sélectionnez que les jetons qui sont au moins autant susceptibles d'apparaître que les jetons aléatoires, étant donné le texte précédent.

### epsilon_cutoff

En unités de 1e-4 ; une valeur raisonnable est 3. Cela définit un plancher de probabilité en dessous duquel les jetons sont exclus de l'échantillonnage. Doit être utilisé avec top_p, top_k, et eta_cutoff fixé à 0.

### eta_cutoff

En unités de 1e-4 ; une valeur raisonnable est 3. Doit être utilisé avec top_p, top_k, et epsilon_cutoff fixé à 0.

### repetition_penalty

Facteur de pénalité exponentielle pour la répétition des jetons précédents. 1 signifie aucune pénalité, valeur plus élevée = moins de répétition, valeur plus faible = plus de répétition.

### repetition_penalty_range

Le nombre de jetons les plus récents à considérer pour la pénalité de répétition. 0 fait que tous les jetons sont utilisés.

### encoder_repetition_penalty

Aussi connu sous le nom de "filtre des hallucinations". Utilisé pour pénaliser les jetons qui *ne sont pas* dans le texte précédent. Valeur plus élevée = plus susceptible de rester dans le contexte, valeur plus faible = plus susceptible de diverger.

### no_repeat_ngram_size

Si ce n'est pas fixé à 0, spécifie la longueur des ensembles de jetons qui sont complètement bloqués de toute répétition. Des valeurs plus élevées = bloquent des phrases plus grandes, des valeurs plus faibles = bloquent des mots ou des lettres de se répéter. Seules les valeurs 0 ou élevées sont recommandées dans la plupart des cas.

### min_length

Longueur minimale de génération en jetons.

### penalty_alpha

La Recherche Contrastée est activée en fixant ceci à plus de zéro et en décochant "do_sample". Elle devrait être utilisée avec une faible valeur de top_k, par exemple, top_k = 4.

