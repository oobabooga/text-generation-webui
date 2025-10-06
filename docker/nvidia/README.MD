Text Generation-Webui-Podman

Updated Dockerfile and docker-compose.yml files for Nvidia, optimized to run with Podman.

Inside the 'docker' folder there is an 'nvidia' folder which contains the files as they are needed to build and run the container using:

sudo podman compose up -d

You will need podman-compose installed and the nvidia-container-toolkit installed with the proper Selinux permissions, I personally use this Fedora Copr repo which installs the nvidia-container-toolkit with the Selinux configuration out of the box.

https://copr.fedorainfracloud.org/coprs/g/ai-ml/nvidia-container-toolkit/

The ports you are going to use must be whitelisted, and also make sure the volumes you are going to use point to existing folders with enough permissions and have the ':Z' tag, besides that it should work out of the box if the permissions are set properly, as the default configuration addresses Podman requirements while maintaining the original.

Enjoy!
