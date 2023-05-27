# RUN INSIDE WSL https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl

sudo apt-key del 7fa2af80 && \
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin && \
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb && \
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb && \
sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
sudo apt-get update && \
sudo apt-get -y install cuda && \
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list && \
sudo apt-get update && \
sudo apt-get -y install nvidia-container-runtime

# THEN UPDATE YOUR DAEMON.JSON https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup
#
#   "default-runtime": "nvidia",
#   "runtimes": {
#     "nvidia": {
#       "path": "/usr/bin/nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
#