# One-click installers

These are automated installers for [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui).

The idea is to allow people to use the program without having to type commands in the terminal, thus making it more accessible.

## How it works

The `start` scripts download miniconda, create a conda environment inside the current folder, and then install the webui using that environment.

After the initial installation, the `update` scripts are then used to automatically pull the latest text-generation-webui code and upgrade its requirements.

## Limitations

* The start/update scripts themselves are not automatically updated. To update them, you have to re-download the zips listed on the [main README](https://github.com/oobabooga/text-generation-webui#one-click-installers) and overwrite your existing files.
