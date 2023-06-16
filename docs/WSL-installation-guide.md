Guide created by [@jfryton](https://github.com/jfryton). Thank you jfryton.

-----

Here's an easy-to-follow, step-by-step guide for installing Windows Subsystem for Linux (WSL) with Ubuntu on Windows 10/11:

## Step 1: Enable WSL

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

## Step 2: Install Ubuntu

1. Open the Microsoft Store.
2. Search for "Ubuntu" in the search bar.
3. Choose the desired Ubuntu version (e.g., Ubuntu 20.04 LTS) and click "Get" or "Install" to download and install the Ubuntu app.
4. Once the installation is complete, click "Launch" or search for "Ubuntu" in the Start menu and open the app.

## Step 3: Set up Ubuntu

1. When you first launch the Ubuntu app, it will take a few minutes to set up. Be patient as it installs the necessary files and sets up your environment.
2. Once the setup is complete, you will be prompted to create a new UNIX username and password. Choose a username and password, and make sure to remember them, as you will need them for future administrative tasks within the Ubuntu environment.

## Step 4: Update and upgrade packages

1. After setting up your username and password, it's a good idea to update and upgrade your Ubuntu system. Run the following commands in the Ubuntu terminal:

```
sudo apt update
sudo apt upgrade
```

2. Enter your password when prompted. This will update the package list and upgrade any outdated packages.

Congratulations! You have now installed WSL with Ubuntu on your Windows 10/11 system. You can use the Ubuntu terminal for various tasks, like running Linux commands, installing packages, or managing files.

You can launch your WSL Ubuntu installation by selecting the Ubuntu app (like any other program installed on your computer) or typing 'ubuntu' into Powershell or Terminal.

## Step 5: Proceed with Linux instructions

1. You can now follow the Linux setup instructions. If you receive any error messages about a missing tool or package, just install them using apt:

```
sudo apt install [missing package]
```

You will probably need to install build-essential

```
sudo apt install build-essential
```

If you face any issues or need to troubleshoot, you can always refer to the official Microsoft documentation for WSL: https://docs.microsoft.com/en-us/windows/wsl/

#### WSL2 performance using /mnt: 
when you git clone a repository, put it inside WSL and not outside. To understand more, take a look at this [issue](https://github.com/microsoft/WSL/issues/4197#issuecomment-604592340)

## Bonus: Port Forwarding

By default, you won't be able to access the webui from another device on your local network. You will need to setup the appropriate port forwarding using the following command (using PowerShell or Terminal with administrator privileges). 

```
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=7860 connectaddress=localhost connectport=7860
```
