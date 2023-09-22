# Installation

Run the "start" script. It will install the web UI and all of its
dependencies inside this folder.

To launch the web UI in the future after it is already installed, run the
"start" script again.

## Updating the web UI

Run the "update" script. It will install the updates only, so it should
be much faster than the initial installation.

## Adding flags like --model, --api, etc

Open the "CMD_FLAGS.txt" file with a text editor, add your flags, and
save the file. For instance, to add the --api flag, change the file
contents to

--api

## Running an interactive shell

Sometimes you may need to install some additional Python package. To do
that, run the "cmd" script and type your commands inside the terminal
window that will appear.

## Using an AMD GPU in Linux

Requires ROCm SDK 5.4.2 or 5.4.3 to be installed. Some systems may also
need: sudo apt-get install libstdc++-12-dev

Edit the "webui.py" script using a text editor and un-comment and
modify the lines near the top of the script according to your setup. In
particular, modify the os.environ["ROCM_PATH"] = '/opt/rocm' line to
point to your ROCm installation.

## WSL special instructions

If you do not have WSL installed, see here:
https://learn.microsoft.com/en-us/windows/wsl/install

If you want to install Linux to a drive other than C
Open powershell and enter these commands:

cd D:\Path\To\Linux
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri <LinuxDistroURL> -OutFile Linux.appx -UseBasicParsing
mv Linux.appx Linux.zip

Then open Linux.zip and you should see several .appx files inside.
The one with _x64.appx contains the exe installer that you need.
Extract the contents of that _x64.appx file and run <distro>.exe to install.

Linux Distro URLs:
https://learn.microsoft.com/en-us/windows/wsl/install-manual#downloading-distributions

******************************************************************************
*ENSURE THAT THE WSL LINUX DISTRO THAT YOU WISH TO USE IS SET AS THE DEFAULT!*
******************************************************************************

Do this by using these commands:
wsl -l
wsl -s <DistroName>

### Web UI Installation

Run the "start" script. By default it will install the web UI in WSL:
/home/{username}/text-gen-install

To launch the web UI in the future after it is already installed, run
the same "start" script. Ensure that webui.py and wsl.sh are next to it!

### Updating the web UI

As an alternative to running the "update" script, you can also run "wsl.sh update" in WSL.

### Running an interactive shell

To run an interactive shell in the miniconda environment, run the "cmd"
script. This is useful for installing additional requirements manually.

As an alternative to running the "cmd" script, you can also run "wsl.sh cmd" in WSL.

### Changing the default install location

To change this, you will need to edit the scripts as follows:
wsl.sh: line ~22   INSTALL_DIR="/path/to/install/dir"

Keep in mind that there is a long-standing bug in WSL that significantly
slows drive read/write speeds when using a physical drive as opposed to
the virtual one that Linux is installed in.
