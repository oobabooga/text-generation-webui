Guide créé par [@jfryton](https://github.com/jfryton). Merci jfryton.

-----

Voici un guide étape par étape, facile à suivre, pour installer le Windows Subsystem for Linux (WSL) avec Ubuntu sur Windows 10/11 :

## Étape 1 : Activer WSL

1. Appuyez sur la touche Windows + X et cliquez sur "Windows PowerShell (Admin)" ou "Windows Terminal (Admin)" pour ouvrir PowerShell ou Terminal avec les privilèges d'administrateur.
2. Dans la fenêtre PowerShell, tapez la commande suivante et appuyez sur Entrée :

```
wsl --install
```

Si cette commande ne fonctionne pas, vous pouvez activer WSL avec la commande suivante pour Windows 10 :

```
wsl --set-default-version 1
```

Pour Windows 11, vous pouvez utiliser :

```
wsl --set-default-version 2
```

Il se peut qu'on vous demande de redémarrer votre ordinateur. Si c'est le cas, enregistrez votre travail et redémarrez.

## Étape 2 : Installer Ubuntu

1. Ouvrez le Microsoft Store.
2. Recherchez "Ubuntu" dans la barre de recherche.
3. Choisissez la version d'Ubuntu souhaitée (par exemple, Ubuntu 20.04 LTS) et cliquez sur "Obtenir" ou "Installer" pour télécharger et installer l'application Ubuntu.
4. Une fois l'installation terminée, cliquez sur "Lancer" ou recherchez "Ubuntu" dans le menu Démarrer et ouvrez l'application.

## Étape 3 : Configurer Ubuntu

1. Lorsque vous lancez l'application Ubuntu pour la première fois, elle prendra quelques minutes pour se configurer. Soyez patient pendant qu'elle installe les fichiers nécessaires et configure votre environnement.
2. Une fois la configuration terminée, on vous demandera de créer un nouveau nom d'utilisateur et un mot de passe UNIX. Choisissez un nom d'utilisateur et un mot de passe, et n'oubliez pas de les noter, car vous en aurez besoin pour les futures tâches administratives dans l'environnement Ubuntu.

## Étape 4 : Mettre à jour et actualiser les paquets

1. Après avoir configuré votre nom d'utilisateur et votre mot de passe, il est recommandé de mettre à jour et d'actualiser votre système Ubuntu. Exécutez les commandes suivantes dans le terminal Ubuntu :

```
sudo apt update
sudo apt upgrade
```

2. Entrez votre mot de passe lorsque cela vous est demandé. Cela mettra à jour la liste des paquets et actualisera tous les paquets obsolètes.

Félicitations ! Vous avez maintenant installé WSL avec Ubuntu sur votre système Windows 10/11. Vous pouvez utiliser le terminal Ubuntu pour diverses tâches, comme exécuter des commandes Linux, installer des paquets ou gérer des fichiers.

Vous pouvez lancer votre installation WSL Ubuntu en sélectionnant l'application Ubuntu (comme n'importe quel autre programme installé sur votre ordinateur) ou en tapant 'ubuntu' dans Powershell ou Terminal.

## Étape 5 : Suivez les instructions pour Linux

1. Vous pouvez maintenant suivre les instructions de configuration pour Linux. Si vous recevez des messages d'erreur concernant un outil ou un paquet manquant, installez-les en utilisant apt :

```
sudo apt install [paquet manquant]
```

Vous devrez probablement installer build-essential

```
sudo apt install build-essential
```

Si vous rencontrez des problèmes ou avez besoin d'aide pour le dépannage, vous pouvez toujours consulter la documentation officielle de Microsoft pour WSL : https://docs.microsoft.com/fr-fr/windows/wsl/

#### Performance WSL2 en utilisant /mnt : 
lorsque vous clonez un dépôt git, placez-le à l'intérieur de WSL et non à l'extérieur. Pour en savoir plus, jetez un œil à ce [problème](https://github.com/microsoft/WSL/issues/4197#issuecomment-604592340)

## Bonus : Redirection de port

Par défaut, vous ne pourrez pas accéder au webui depuis un autre appareil sur votre réseau local. Vous devrez configurer la redirection de port appropriée en utilisant la commande suivante (en utilisant PowerShell ou Terminal avec les droits d'administrateur).

```
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7860 connectaddress=localhost connectport=7860
```
