- [Linux](#linux)
  - [Ubuntu 22.04](#ubuntu-2204)
    - [update the drivers](#update-the-drivers)
    - [reboot](#reboot)
    - [docker \& container toolkit](#docker--container-toolkit)
  - [Manjaro](#manjaro)
    - [update the drivers](#update-the-drivers-1)
    - [reboot](#reboot-1)
    - [docker \& container toolkit](#docker--container-toolkit-1)
  - [prepare environment \& startup](#prepare-environment--startup)
    - [place models in models folder](#place-models-in-models-folder)
    - [prepare .env file](#prepare-env-file)
    - [startup docker container](#startup-docker-container)
- [Windows](#windows)
# Linux

## Ubuntu 22.04

### update the drivers
in the the “software updater” update drivers to the last version of the prop driver.

### reboot
to switch using to new driver

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

### docker & container toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia.list > /dev/null 

sudo apt update

sudo apt install nvidia-docker2 nvidia-container-runtime -y
sudo systemctl restart docker
```

## Manjaro

### update the drivers
```bash
sudo mhwd -a pci nonfree 0300
```
### reboot
```bash
reboot
```
### docker & container toolkit
```bash
yay -S docker docker-compose buildkit gcc nvidia-docker
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl restart docker # required by nvidia-container-runtime
```

## prepare environment & startup

### place models in models folder
download and place the models inside the models folder. tested with:

4bit
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483891617
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1483941105

8bit:
https://github.com/oobabooga/text-generation-webui/pull/530#issuecomment-1484235789

### prepare .env file
edit .env values to your needs
```bash
cp .env.example .env
nano .env
```

### startup docker container
```bash
docker-compose up --build
```


# Windows
coming soon