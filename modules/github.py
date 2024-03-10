import subprocess
from pathlib import Path

new_extensions = set()


def clone_or_pull_repository(github_url):
    global new_extensions

    repository_folder = Path("extensions")
    repo_name = github_url.rstrip("/").split("/")[-1].split(".")[0]

    # Check if the repository folder exists
    if not repository_folder.exists():
        repository_folder.mkdir(parents=True)

    repo_path = repository_folder / repo_name

    # Check if the repository is already cloned
    if repo_path.exists():
        yield f"正在更新 {github_url}..."
        # Perform a 'git pull' to update the repository
        try:
            pull_output = subprocess.check_output(["git", "-C", repo_path, "pull"], stderr=subprocess.STDOUT)
            yield "已完成。"
            return pull_output.decode()
        except subprocess.CalledProcessError as e:
            return str(e)

    # Clone the repository
    try:
        yield f"正在克隆 {github_url}..."
        clone_output = subprocess.check_output(["git", "clone", github_url, repo_path], stderr=subprocess.STDOUT)
        new_extensions.add(repo_name)
        yield f"扩展 `{repo_name}` 已下载。\n\n请完全关闭网页界面并重新启动，以便能够加载它。"
        return clone_output.decode()
    except subprocess.CalledProcessError as e:
        return str(e)