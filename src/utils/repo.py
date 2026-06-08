"""Walk a repo, yield indexable file paths.

Filters out:
- Hidden dirs (.git, .venv, .vscode, ...)
- Common build/cache dirs (__pycache__, node_modules, target, build, dist)
- Files whose extension isn't in any LanguageAdapter's registry
- Files larger than MAX_FILE_BYTES (likely generated/binary)
- Symlinks (avoid loops)

Used by builder to feed Chunker.
"""

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

    for path in root.rglob("*"):
        # Skip if any parent dir is in IGNORE_DIRS or starts with .
        if any(p.name in IGNORE_DIRS or p.name.startswith(".") for p in path.parents):
            continue

        # Skip if the file itself starts with .
        if path.name.startswith("."):
            continue

        # Skip symlinks
        if path.is_symlink():
            continue

        # Skip directories
        if not path.is_file():
            continue

        # Skip if too large
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        # Skip if no adapter for this extension
        if get_adapter(path) is None:
            continue

        yield path

