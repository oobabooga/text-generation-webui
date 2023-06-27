import os
import subprocess


def clone_or_pull_repository(github_url):
    repository_folder = "extensions"
    repo_name = github_url.split("/")[-1].split(".")[0]

    # Check if the repository folder exists
    if not os.path.exists(repository_folder):
        os.makedirs(repository_folder)

    repo_path = os.path.join(repository_folder, repo_name)

    # Check if the repository is already cloned
    if os.path.exists(repo_path):
        # Perform a 'git pull' to update the repository
        try:
            pull_output = subprocess.check_output(["git", "-C", repo_path, "pull"], stderr=subprocess.STDOUT)
            return pull_output.decode()
        except subprocess.CalledProcessError as e:
            return str(e)

    # Clone the repository
    try:
        clone_output = subprocess.check_output(["git", "clone", github_url, repo_path], stderr=subprocess.STDOUT)
        return clone_output.decode()
    except subprocess.CalledProcessError as e:
        return str(e)
