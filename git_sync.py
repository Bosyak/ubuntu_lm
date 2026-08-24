"""
Клонирует репозиторий при первом запуске, либо делает pull если он уже есть.
"""
import os
from git import Repo, GitCommandError

import config


def sync_repo() -> str:
    """
    Возвращает путь к локальной папке репозитория.
    Если папки нет - клонирует. Если есть - делает pull.
    """
    repo_path = config.REPO_PATH

    if os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"[git_sync] Репозиторий уже есть в {repo_path}, делаю pull...")
        repo = Repo(repo_path)
        try:
            origin = repo.remotes.origin
            origin.pull(config.REPO_BRANCH)
        except GitCommandError as e:
            print(f"[git_sync] Не удалось сделать pull: {e}")
            raise
    else:
        print(f"[git_sync] Клонирую {config.REPO_URL} в {repo_path}...")
        os.makedirs(os.path.dirname(repo_path.rstrip("/")) or ".", exist_ok=True)
        Repo.clone_from(config.REPO_URL, repo_path, branch=config.REPO_BRANCH)

    return repo_path


if __name__ == "__main__":
    path = sync_repo()
    print(f"Готово. Репозиторий лежит в: {path}")
