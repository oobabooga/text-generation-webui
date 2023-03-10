# Installation instructions

- On all platforms, run `pip install -r requirements.txt` in this folder
- You need **PortAudio** to run the speech recognition. Below are guides for all platforms


## Windows

- You don't need to do anything, `pyaudio` already comes with PortAudio included on Windows.

## Mac

```commandline
brew install portaudio
brew link --overwrite portaudio
pip install pyaudio
```

## Linux

- You have to use your distro's package manager to install PortAudio.

### Ubuntu / Debian / Mint

```commandline
sudo apt install portaudio19-dev python3-pyaudio
```

### Arch / Manjaro

```commandline
sudo pacman -S portaudio
```

### Fedora

```commandline
sudo dnf -y install portaudio
```