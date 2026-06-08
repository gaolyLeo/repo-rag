"""Walk a repo, yield indexable file paths.

Prefers `git ls-files` when the directory is a git repo — this naturally
respects .gitignore, .gitattributes, and the global gitignore. Falls back
to a manual walk for non-git directories.

Filters out:
- Hidden dirs (.git, .venv, .vscode, ...)
- Common build/cache dirs (__pycache__, node_modules, target, build, dist)
- Files whose extension isn't in any LanguageAdapter's registry
- Files larger than MAX_FILE_BYTES (likely generated/binary)
- Symlinks (avoid loops)

Used by builder to feed Chunker.
"""

import subprocess
from pathlib import Path
from typing import Iterator

from chunking.language import get_adapter


IGNORE_DIRS: set[str] = {
    "__pycache__", "node_modules", "target", "build", "dist",
    ".git", ".venv", "venv", ".vscode", ".idea",
}

MAX_FILE_BYTES: int = 1_000_000  # 1 MB


def iter_files(repo_path: str) -> Iterator[Path]:
    """Yield files under repo_path that we should index."""
    root = Path(repo_path)

    if _is_git_repo(root):
        yield from _iter_git_files(root)
    else:
        yield from _iter_walk_files(root)


def _is_git_repo(root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root, capture_output=True,
    )
    return result.returncode == 0


def _iter_git_files(root: Path) -> Iterator[Path]:
    """Use `git ls-files` — respects .gitignore automatically."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root, capture_output=True,
    )
    for rel in result.stdout.split(b"\x00"):
        if not rel:
            continue
        path = root / rel.decode()
        if not path.is_file() or path.is_symlink():
            continue
        if get_adapter(path) is None:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _iter_walk_files(root: Path) -> Iterator[Path]:
    """Fallback: manual walk with hardcoded ignore list."""
    for path in root.rglob("*"):
        if any(p.name in IGNORE_DIRS or p.name.startswith(".") for p in path.parents):
            continue
        if path.name.startswith("."):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if get_adapter(path) is None:
            continue
        yield path

