## Personnages de chat

Les personnages personnalisés pour le mode de chat sont définis par des fichiers `.yaml` à l'intérieur du dossier `characters`. Un exemple est inclus : [Example.yaml](https://github.com/oobabooga/text-generation-webui/blob/main/characters/Example.yaml).

Les champs suivants peuvent être définis :

| Champ | Description |
|-------|-------------|
| `name` ou `bot` | Le nom du personnage. |
| `context` | Une chaîne de caractères qui apparaît en haut de l'invite. Elle contient généralement une description de la personnalité du personnage et quelques messages exemples. |
| `greeting` (facultatif) | Le message d'accueil du personnage. Il apparaît lorsque le personnage est chargé pour la première fois ou lorsque l'historique est effacé. |
| `your_name` ou `user` (facultatif) | Votre nom. Cela écrase ce que vous aviez précédemment écrit dans le champ `Votre nom` dans l'interface. |

#### Tokens spéciaux

Les remplacements suivants se produisent lorsque l'invite est générée, et ils s'appliquent aux champs `context` et `greeting` :

* `{{char}}` et `<BOT>` sont remplacés par le nom du personnage.
* `{{user}}` et `<USER>` sont remplacés par votre nom.

#### Comment ajouter une photo de profil pour mon personnage ?

Placez une image portant le même nom que le fichier `.yaml` de votre personnage dans le dossier `characters`. Par exemple, si votre bot est `Character.yaml`, ajoutez `Character.jpg` ou `Character.png` au dossier.

#### L'historique du chat est-il tronqué dans l'invite ?

Lorsque votre invite atteint le paramètre `truncation_length` (2048 par défaut), les anciens messages sont supprimés un par un. La chaîne de contexte restera toujours en haut de l'invite et ne sera jamais tronquée.

## Styles de chat

Des styles de chat personnalisés peuvent être définis dans le dossier `text-generation-webui/css`. Créez simplement un nouveau fichier dont le nom commence par `chat_style-` et se termine par `.css` et il apparaîtra automatiquement dans le menu déroulant "Style de chat" de l'interface. Exemples :

```
chat_style-cai-chat.css
chat_style-TheEncrypted777.css
chat_style-wpp.css
```

Vous devriez utiliser les mêmes noms de classe que dans `chat_style-cai-chat.css` pour votre style personnalisé.
