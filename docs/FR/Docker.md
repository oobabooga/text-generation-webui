Docker Compose est une méthode pour installer et lancer l'interface web UI dans une image isolée d'Ubuntu en utilisant seulement quelques commandes.

Pour créer l'image comme décrit dans le README principal, vous devez avoir Docker Compose version 2.17 ou plus :

```
~$ docker compose version
Docker Compose version v2.17.2
```

Assurez-vous également de créer les liens symboliques nécessaires :

```
cd text-generation-webui
ln -s docker/{Dockerfile,docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
# Modifier .env et définir TORCH_CUDA_ARCH_LIST en fonction de votre modèle GPU
docker compose up --build
```

# Table des matières

* [Instructions d'installation de Docker Compose](#docker-compose-installation-instructions)
* [Répertoire avec des fichiers Docker supplémentaires](#dedicated-docker-repository)

# Instructions d'installation de Docker Compose

Par [@loeken](https://github.com/loeken).

- [Table des matières](#table-des-matières)
- [Instructions d'installation de Docker Compose](#instructions-dinstallation-de-docker-compose)
  - [Ubuntu 22.04](#ubuntu-2204)
    - [0. vidéo YouTube](#0-vidéo-youtube)
    - [1. mettre à jour les pilotes](#1-mettre-à-jour-les-pilotes)
    - [2. redémarrage](#2-redémarrage)
    - [3. installer Docker](#3-installer-docker)
    - [4. Docker \& kit d'outils pour conteneurs](#4-docker--kit-doutils-pour-conteneurs)
    - [5. cloner le dépôt](#5-cloner-le-dépôt)
    - [6. préparer les modèles](#6-préparer-les-modèles)
    - [7. préparer le fichier .env](#7-préparer-le-fichier-env)
    - [8. démarrer le conteneur Docker](#8-démarrer-le-conteneur-docker)
  - [Manjaro](#manjaro)
    - [mettre à jour les pilotes](#mettre-à-jour-les-pilotes)
    - [redémarrage](#redémarrage)
    - [Docker \& kit d'outils pour conteneurs](#docker--kit-doutils-pour-conteneurs)
    - [continuer avec la tâche d'Ubuntu](#continuer-avec-la-tâche-dubuntu)
  - [Windows](#windows)
    - [0. vidéo YouTube](#0-vidéo-youtube-1)
    - [1. gestionnaire de paquets choco](#1-gestionnaire-de-paquets-choco)
    - [2. installer les pilotes/dépendances](#2-installer-les-pilotesdépendances)
    - [3. Installez WSL](#3-installez-wsl)
    - [4. Redémarrez](#4-redémarrez)
    - [5. Git clone et démarrage](#5-git-clone-et-démarrage)
    - [6. Préparez les modèles](#6-préparez-les-modèles)
    - [7. Démarrage](#7-démarrage)
  - [Notes](#notes)
- [Dépôt Docker dédié](#dépôt-docker-dédié)

## Ubuntu 22.04

### 0. vidéo YouTube
Une vidéo vous guidant à travers l'installation peut être trouvée ici :

[![oobabooga text-generation-webui installation dans Docker sur Ubuntu 22.04](https://img.youtube.com/vi/ELkKWYh8qOk/0.jpg)](https://www.youtube.com/watch?v=ELkKWYh8qOk)

### 1. mettre à jour les pilotes
Dans le “gestionnaire de mises à jour”, mettez à jour les pilotes à la dernière version du pilote propriétaire.

### 2. redémarrage
Pour passer à l'utilisation du nouveau pilote.

### 3. installer Docker
```bash
sudo apt update
sudo apt-get install curl
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-compose -y
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Docker & kit d'outils pour conteneurs
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia.list > /dev/null 
sudo apt update
sudo apt install nvidia-docker2 nvidia-container-runtime -y
sudo systemctl restart docker
```


### 5. cloner le dépôt
```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
```

### 6. préparer les modèles
Téléchargez et placez les modèles dans le dossier des modèles. Testé avec :

4bit
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105

8bit:
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789

### 7. préparer le fichier .env
Modifiez les valeurs de .env selon vos besoins.
```bash
cp .env.example .env
nano .env
```

### 8. démarrer le conteneur Docker
```bash
docker compose up --build
```

## Manjaro
Manjaro/Arch est similaire à Ubuntu, mais l'installation des dépendances est plus pratique.

### mettre à jour les pilotes
```bash
sudo mhwd -a pci nonfree 0300
```

### redémarrage
```bash
reboot
```

### Docker & kit d'outils pour conteneurs
```bash
yay -S docker docker-compose buildkit gcc nvidia-docker
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl restart docker # requis par nvidia-container-runtime
```

### continuer avec la tâche d'Ubuntu
Poursuivez à [5. cloner le dépôt](#5-cloner-le-dépôt)


## Windows
### 0. vidéo YouTube
Une vidéo vous guidant à travers l'installation peut être trouvée ici :
[![oobabooga text-generation-webui installation dans Docker sur Windows 11](https://img.youtube.com/vi/ejH4w5b5kFQ/0.jpg)](https://www.youtube.com/watch?v=ejH4w5b5kFQ)

### 1. gestionnaire de paquets choco
Installez le gestionnaire de paquets (https://chocolatey.org/ )
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 2. installer les pilotes/dépendances
```
choco install nvidia-display-driver cuda git docker-desktop
``` 

### 3. Installez WSL
wsl --install

### 4. Redémarrez
Après le redémarrage, entrez le nom d'utilisateur et le mot de passe dans WSL.

### 5. Git clone et démarrage
Clonez le dépôt et modifiez les valeurs de .env selon vos besoins.
```
cd Desktop
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
COPY .env.example .env
notepad .env
```

### 6. Préparez les modèles
Téléchargez et placez les modèles dans le dossier des modèles. Testé avec :

4bit https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617 https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105

8bit: https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789

### 7. Démarrage
```
docker compose up
```

## Notes

Sur les anciennes versions d'Ubuntu, vous pouvez manuellement installer le plugin Docker Compose de cette manière :
```
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.17.2/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
export PATH="$HOME/.docker/cli-plugins:$PATH"
```

# Dépôt Docker dédié

Un dépôt externe maintient un wrapper Docker pour ce projet ainsi que plusieurs variantes 'un clic' de `docker compose` pré-configurées (par exemple, des branches mises à jour de GPTQ). Il est disponible ici : [Atinoda/text-generation-webui-docker](https://github.com/Atinoda/text-generation-webui-docker).

