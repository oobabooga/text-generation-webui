import argparse
import glob
import os
import subprocess
import site
import sys

script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

# Remove the '# ' from the following lines as needed for your AMD GPU on Linux
# os.environ["ROCM_PATH"] = '/opt/rocm'
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = '10.3.0'
# os.environ["HCC_AMDGPU_TARGET"] = 'gfx1030'

# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
    with open(cmd_flags_path, 'r') as f:
        CMD_FLAGS = ' '.join(line.strip() for line in f if line.strip() and not line.strip().startswith('#'))
else:
    CMD_FLAGS = ''


def is_linux():
    return sys.platform.startswith("linux")


def is_windows():
    return sys.platform.startswith("win")


def is_macos():
    return sys.platform.startswith("darwin")


def is_installed():
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        return os.path.isfile(os.path.join(site_packages_path, 'torch', '__init__.py'))
    else:
        return os.path.isdir(conda_env_path)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def clear_cache():
    run_cmd("conda clean -a -y", environment=True)
    run_cmd("python -m pip cache purge", environment=True)


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        if line.strip() != '':
            print("*", line)

    print("*******************************************************************\n\n")


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = "\"" + conda_bat_path + "\" activate \"" + conda_env_path + "\" >nul && " + cmd
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = ". \"" + conda_sh_path + "\" && conda activate \"" + conda_env_path + "\" && " + cmd

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print("Command '" + cmd + "' failed with exit status code '" + str(result.returncode) + "'. Exiting...")
        sys.exit()

    return result


def install_webui():

    # Select your GPU, or choose to run in CPU mode
    if "GPU_CHOICE" in os.environ:
        choice = os.environ["GPU_CHOICE"].upper()
        print_big_message(f"Selected GPU choice \"{choice}\" based on the GPU_CHOICE environment variable.")
    else:
        print("What is your GPU")
        print()
        print("A) NVIDIA")
        print("B) AMD (Linux/MacOS only. Requires ROCm SDK 5.4.2/5.4.3 on Linux)")
        print("C) Apple M Series")
        print("D) None (I want to run models in CPU mode)")
        print()

        choice = input("Input> ").upper()
        while choice not in ['A', 'B', 'C', 'D']:
            print("Invalid choice. Please try again.")
            choice = input("Input> ").upper()

    if choice == "D":
        print_big_message("Once the installation ends, make sure to open CMD_FLAGS.txt with\na text editor and add the --cpu flag.")

    # Find the proper Pytorch installation command
    install_git = "conda install -y -k ninja git"
    install_pytorch = "python -m pip install torch torchvision torchaudio"

    if is_windows() and choice == "A":
        install_pytorch = "python -m pip install torch==2.0.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
    elif not is_macos() and choice == "B":
        if is_linux():
            install_pytorch = "python -m pip install torch==2.0.1+rocm5.4.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2"
        else:
            print("AMD GPUs are only supported on Linux. Exiting...")
            sys.exit()
    elif is_linux() and (choice == "C" or choice == "D"):
        install_pytorch = "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    # Install Git and then Pytorch
    run_cmd(f"{install_git} && {install_pytorch}", assert_success=True, environment=True)

    # Install the webui requirements
    update_requirements(initial_installation=True)


def update_requirements(initial_installation=False):

    # Create .git directory if missing
    if not os.path.isdir(os.path.join(script_dir, ".git")):
        git_creation_cmd = 'git init -b main && git remote add origin https://github.com/oobabooga/text-generation-webui && git fetch && git remote set-head origin -a && git reset origin/HEAD && git branch --set-upstream-to=origin/HEAD'
        run_cmd(git_creation_cmd, environment=True, assert_success=True)

    run_cmd("git pull --autostash", assert_success=True, environment=True)

    # Initial installation only: install the extensions requirements
    if initial_installation:
        extensions = next(os.walk("extensions"))[1]
        for extension in extensions:
            if extension in ['superbooga']:  # No wheels available for requirements
                continue

            extension_req_path = os.path.join("extensions", extension, "requirements.txt")
            if os.path.exists(extension_req_path):
                run_cmd("python -m pip install -r " + extension_req_path + " --upgrade", assert_success=True, environment=True)

    textgen_requirements = open("requirements.txt").read().splitlines()

    # Workaround for git+ packages not updating properly. Also store requirements.txt for later use
    git_requirements = [req for req in textgen_requirements if req.startswith("git+")]

    # Loop through each "git+" requirement and uninstall it
    for req in git_requirements:
        # Extract the package name from the "git+" requirement
        url = req.replace("git+", "")
        package_name = url.split("/")[-1].split("@")[0]

        # Uninstall the package using pip
        run_cmd("python -m pip uninstall -y " + package_name, environment=True)
        print(f"Uninstalled {package_name}")

    # Install/update the project requirements
    run_cmd("python -m pip install -r requirements.txt --upgrade", assert_success=True, environment=True)

    # The following requirements are for CUDA, not CPU
    # Parse output of 'pip show torch' to determine torch version
    torver_cmd = run_cmd("python -m pip show torch", assert_success=True, environment=True, capture_output=True)
    torver = [v.split()[1] for v in torver_cmd.stdout.decode('utf-8').splitlines() if 'Version:' in v][0]

    # Check for '+cu' or '+rocm' in version string to determine if torch uses CUDA or ROCm. Check for pytorch-cuda as well for backwards compatibility
    if '+cu' not in torver and '+rocm' not in torver and run_cmd("conda list -f pytorch-cuda | grep pytorch-cuda", environment=True, capture_output=True).returncode == 1:
        clear_cache()
        return

    if not os.path.exists("repositories/"):
        os.mkdir("repositories")

    os.chdir("repositories")

    # Install or update ExLlama as needed
    if not os.path.exists("exllama/"):
        run_cmd("git clone https://github.com/turboderp/exllama.git", environment=True)
    else:
        os.chdir("exllama")
        run_cmd("git pull", environment=True)
        os.chdir("..")

    # Pre-installed ExLlama module does not support AMD GPU
    if '+rocm' in torver:
        run_cmd("python -m pip uninstall -y exllama", environment=True)
        # Get download URL for latest ExLlama ROCm wheel
        exllama_rocm = run_cmd('curl -s https://api.github.com/repos/jllllll/exllama/releases/latest | grep browser_download_url | grep rocm5.4.2-cp310-cp310-linux_x86_64.whl | cut -d : -f 2,3 | tr -d \'"\'', environment=True, capture_output=True).stdout.decode('utf-8')
        if 'rocm5.4.2-cp310-cp310-linux_x86_64.whl' in exllama_rocm:
            run_cmd("python -m pip install " + exllama_rocm, environment=True)

    # Fix JIT compile issue with ExLlama in Linux/WSL
    if is_linux() and not os.path.exists(f"{conda_env_path}/lib64"):
        run_cmd(f'ln -s "{conda_env_path}/lib" "{conda_env_path}/lib64"', environment=True)

    # On some Linux distributions, g++ may not exist or be the wrong version to compile GPTQ-for-LLaMa
    if is_linux():
        gxx_output = run_cmd("g++ -dumpfullversion -dumpversion", environment=True, capture_output=True)
        if gxx_output.returncode != 0 or int(gxx_output.stdout.strip().split(b".")[0]) > 11:
            # Install the correct version of g++
            run_cmd("conda install -y -k conda-forge::gxx_linux-64=11.2.0", environment=True)

    # Install/Update ROCm AutoGPTQ for AMD GPUs
    if '+rocm' in torver:
        auto_gptq_version = [req for req in textgen_requirements if req.startswith('https://github.com/PanQiWei/AutoGPTQ/releases/download/')][0].split('/')[7]
        auto_gptq_wheel = run_cmd(f'curl -s https://api.github.com/repos/PanQiWei/AutoGPTQ/releases/tags/{auto_gptq_version} | grep browser_download_url | grep rocm5.4.2-cp310-cp310-linux_x86_64.whl | cut -d : -f 2,3 | tr -d \'"\'', environment=True, capture_output=True).stdout.decode('utf-8')
        if not auto_gptq_wheel and run_cmd(f"python -m pip install {auto_gptq_wheel} --force-reinstall --no-deps", environment=True).returncode != 0:
            print_big_message("ERROR: AutoGPTQ wheel installation failed!\n       You will not be able to use GPTQ-based models with AutoGPTQ.")

    # Install GPTQ-for-LLaMa for ROCm
    if '+rocm' in torver:
        gptq_wheel = run_cmd('curl -s https://api.github.com/repos/jllllll/GPTQ-for-LLaMa-CUDA/releases/latest | grep browser_download_url | grep rocm5.4.2-cp310-cp310-linux_x86_64.whl | cut -d : -f 2,3 | tr -d \'"\'', environment=True, capture_output=True).stdout.decode('utf-8')
        install_gptq = run_cmd("python -m pip install " + gptq_wheel, environment=True).returncode == 0
        if install_gptq:
            print("Wheel installation success!")
        else:
            print("ERROR: GPTQ wheel installation failed.")
            print("You will not be able to use GPTQ-based models with GPTQ-for-LLaMa.")

    clear_cache()


def download_model():
    run_cmd("python download-model.py", environment=True)


def launch_webui():
    flags = [flag for flag in sys.argv[1:] if flag != '--update']
    run_cmd(f"python server.py {' '.join(flags)} {CMD_FLAGS}", environment=True)


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='Update the web UI.')
    args, _ = parser.parse_known_args()

    if args.update:
        update_requirements()
    else:
        # If webui has already been installed, skip and run
        if not is_installed():
            install_webui()
            os.chdir(script_dir)

        if os.environ.get("LAUNCH_AFTER_INSTALL", "").lower() in ("no", "n", "false", "0", "f", "off"):
            print_big_message("Install finished successfully and will now exit due to LAUNCH_AFTER_INSTALL.")
            sys.exit()

        # Check if a model has been downloaded yet
        if len([item for item in glob.glob('models/*') if not item.endswith(('.txt', '.yaml'))]) == 0:
            print_big_message("WARNING: You haven't downloaded any model yet.\nOnce the web UI launches, head over to the \"Model\" tab and download one.")

        # Workaround for llama-cpp-python loading paths in CUDA env vars even if they do not exist
        conda_path_bin = os.path.join(conda_env_path, "bin")
        if not os.path.exists(conda_path_bin):
            os.mkdir(conda_path_bin)

        # Launch the webui
        launch_webui()
