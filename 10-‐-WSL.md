## WSL instructions

If you do not have WSL installed, follow the [instructions below](https://github.com/oobabooga/text-generation-webui/wiki/10-%E2%80%90-WSL#wsl-installation) first.

### Additional WSL setup info

If you want to install Linux to a drive other than C, open powershell and enter these commands:

```
cd D:\Path\To\Linux
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri <LinuxDistroURL> -OutFile Linux.appx -UseBasicParsing
mv Linux.appx Linux.zip
```

Then open Linux.zip and you should see several .appx files inside.

The one with _x64.appx contains the exe installer that you need.

Extract the contents of that _x64.appx file and run <distro>.exe to install.

Linux Distro URLs: https://learn.microsoft.com/en-us/windows/wsl/install-manual#downloading-distributions

**ENSURE THAT THE WSL LINUX DISTRO THAT YOU WISH TO USE IS SET AS THE DEFAULT!**

Do this by using these commands:

```
wsl -l
wsl -s <DistroName>
```

### Web UI Installation

Run the "start" script. By default it will install the web UI in WSL:
/home/{username}/text-gen-install

To launch the web UI in the future after it is already installed, run
the same "start" script. Ensure that one_click.py and wsl.sh are next to it!

### Updating the web UI

As an alternative to running the "update" script, you can also run "wsl.sh update" in WSL.

### Running an interactive shell

As an alternative to running the "cmd" script, you can also run "wsl.sh cmd" in WSL.

### Changing the default install location

To change this, you will need to edit the scripts as follows:
wsl.sh: line ~22   INSTALL_DIR="/path/to/install/dir"

Keep in mind that there is a long-standing bug in WSL that significantly
slows drive read/write speeds when using a physical drive as opposed to
the virtual one that Linux is installed in.

## WSL installation

Guide created by [@jfryton](https://github.com/jfryton). Thank you jfryton.

-----

Here's an easy-to-follow, step-by-step guide for installing Windows Subsystem for Linux (WSL) with Ubuntu on Windows 10/11:

### Step 1: Enable WSL

1. Press the Windows key + X and click on "Windows PowerShell (Admin)" or "Windows Terminal (Admin)" to open PowerShell or Terminal with administrator privileges.
2. In the PowerShell window, type the following command and press Enter:

```
wsl --install
```

If this command doesn't work, you can enable WSL with the following command for Windows 10:

```
wsl --set-default-version 1
```

For Windows 11, you can use:

```
wsl --set-default-version 2
```

You may be prompted to restart your computer. If so, save your work and restart.

### Step 2: Install Ubuntu

1. Open the Microsoft Store.
2. Search for "Ubuntu" in the search bar.
3. Choose the desired Ubuntu version (e.g., Ubuntu 20.04 LTS) and click "Get" or "Install" to download and install the Ubuntu app.
4. Once the installation is complete, click "Launch" or search for "Ubuntu" in the Start menu and open the app.

### Step 3: Set up Ubuntu

1. When you first launch the Ubuntu app, it will take a few minutes to set up. Be patient as it installs the necessary files and sets up your environment.
2. Once the setup is complete, you will be prompted to create a new UNIX username and password. Choose a username and password, and make sure to remember them, as you will need them for future administrative tasks within the Ubuntu environment.

### Step 4: Update and upgrade packages

1. After setting up your username and password, it's a good idea to update and upgrade your Ubuntu system. Run the following commands in the Ubuntu terminal:

```
sudo apt update
sudo apt upgrade
```

2. Enter your password when prompted. This will update the package list and upgrade any outdated packages.

Congratulations! You have now installed WSL with Ubuntu on your Windows 10/11 system. You can use the Ubuntu terminal for various tasks, like running Linux commands, installing packages, or managing files.

You can launch your WSL Ubuntu installation by selecting the Ubuntu app (like any other program installed on your computer) or typing 'ubuntu' into Powershell or Terminal.

### Step 5: Proceed with Linux instructions

1. You can now follow the Linux setup instructions. If you receive any error messages about a missing tool or package, just install them using apt:

```
sudo apt install [missing package]
```

You will probably need to install build-essential

```
sudo apt install build-essential
```

If you face any issues or need to troubleshoot, you can always refer to the official Microsoft documentation for WSL: https://docs.microsoft.com/en-us/windows/wsl/

### WSL2 performance using /mnt: 

When you git clone a repository, put it inside WSL and not outside. To understand more, take a look at this [issue](https://github.com/microsoft/WSL/issues/4197#issuecomment-604592340)

### Bonus: Port Forwarding

By default, you won't be able to access the webui from another device on your local network. You will need to setup the appropriate port forwarding using the following command (using PowerShell or Terminal with administrator privileges). 

```
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7860 connectaddress=localhost connectport=7860
```

