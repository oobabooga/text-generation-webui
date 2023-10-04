# Informations supplémentaires sur les installateurs en un clic

## Installer nvcc

Si vous avez un GPU NVIDIA et que vous avez besoin de compiler quelque chose, comme ExLlamav2 (qui n'a actuellement pas de roues pré-construites), vous pouvez installer `nvcc` en exécutant le script `cmd_` pour votre système d'exploitation et en entrant cette commande :

```
conda install cuda -c nvidia/label/cuda-11.7.1
```

## Utilisation d'un GPU AMD sous Linux

Nécessite l'installation du SDK ROCm 5.4.2 ou 5.4.3. Certains systèmes peuvent également nécessiter : sudo apt-get install libstdc++-12-dev

Modifiez le script "one_click.py" à l'aide d'un éditeur de texte et dé-commentez et modifiez les lignes près du haut du script selon votre configuration. En particulier, modifiez la ligne os.environ["ROCM_PATH"] = '/opt/rocm' pour pointer vers votre installation ROCm.

## Instructions WSL

Si vous n'avez pas WSL installé, voyez ici :
https://learn.microsoft.com/en-us/windows/wsl/install

Si vous voulez installer Linux sur un disque autre que C
Ouvrez powershell et entrez ces commandes :

cd D:\Chemin\Vers\Linux
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri <URLDistroLinux> -OutFile Linux.appx -UseBasicParsing
mv Linux.appx Linux.zip

Ensuite, ouvrez Linux.zip et vous devriez voir plusieurs fichiers .appx à l'intérieur.
Celui avec _x64.appx contient l'installateur exe dont vous avez besoin.
Extrayez le contenu de ce fichier _x64.appx et exécutez <distro>.exe pour installer.

URLs des distributions Linux :
https://learn.microsoft.com/en-us/windows/wsl/install-manual#downloading-distributions

******************************************************************************
*ASSUREZ-VOUS QUE LA DISTRIBUTION LINUX WSL QUE VOUS SOUHAITEZ UTILISER EST DÉFINIE PAR DÉFAUT !*
******************************************************************************

Faites-le en utilisant ces commandes :
wsl -l
wsl -s <NomDistro>

### Installation de l'interface Web

Exécutez le script "start". Par défaut, il installera l'interface web dans WSL :
/home/{nom d'utilisateur}/text-gen-install

Pour lancer l'interface web à l'avenir une fois qu'elle est déjà installée, exécutez le même script "start". Assurez-vous que one_click.py et wsl.sh sont à côté !

### Mise à jour de l'interface Web

Comme alternative à l'exécution du script "update", vous pouvez également exécuter "wsl.sh update" dans WSL.

### Exécution d'un shell interactif

Comme alternative à l'exécution du script "cmd", vous pouvez également exécuter "wsl.sh cmd" dans WSL.

### Modifier l'emplacement d'installation par défaut

Pour ce faire, vous devrez modifier les scripts comme suit :
wsl.sh : ligne ~22   INSTALL_DIR="/chemin/vers/le/répertoire/d'installation"

Gardez à l'esprit qu'il existe un bug de longue date dans WSL qui ralentit considérablement les vitesses de lecture/écriture du disque lors de l'utilisation d'un disque physique par rapport au virtuel sur lequel Linux est installé.