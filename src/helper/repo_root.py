"""Git repository-root detection anchored on a *caller's* source file.

The output-dump layout is built from the repository that owns the decorated
function's file, never from this library's own install directory and never from
the process working directory. That is what makes a PyPI install of
br-py-duckdb-cache write its dumps into the importing project: detection starts
at ``/home/user/my_project/src/app/model.py`` and resolves ``my_project``,
because ``site-packages`` (where this module lives) is not a repository.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

GIT_MARKER_NAME = ".git"


def find_repo_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` (a file or a directory) holding a ``.git`` marker.

    ``.git`` is a directory in a regular clone and a file in a worktree or a
    submodule, so both spellings are accepted. Returns ``None`` when no Git
    repository owns ``start``. Results are cached: the same file is dumped on
    every call, and the answer cannot change while the process runs.
    """
    return _find_repo_root(start.resolve())


@lru_cache(maxsize=512)
def _find_repo_root(resolved: Path) -> Path | None:
    directory = resolved if resolved.is_dir() else resolved.parent
    for candidate in (directory, *directory.parents):
        if (candidate / GIT_MARKER_NAME).exists():
            return candidate
    return None


def find_repo_root_or_parent(start: Path) -> Path:
    """Git root owning ``start``, falling back to ``start``'s parent directory.

    The fallback keeps dumping working for files that live outside any
    repository (a bare script, a temp directory): the file's own directory
    plays the role of the root, which still keeps the layout independent of the
    process working directory.
    """
    return find_repo_root(start) or start.resolve().parent
