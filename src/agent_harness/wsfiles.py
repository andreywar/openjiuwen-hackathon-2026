"""Which files of a task the harness (and the model) may look at.

A task directory holds ``task.json`` - the service file with the statement and,
in the public examples, grading metadata - next to the data of the task. The
data normally sits in ``workspace/``; when that sub-directory is absent the
task directory itself is the workspace. In both layouts:

* the service file ``task.json`` and earlier ``answer.txt`` files at the top
  level of the workspace are never listed, previewed or copied - neither under
  their own name nor through an alias (a symbolic or hard link with another
  name, a nested link, a chain of links): what counts is the object a path
  finally refers to. Links to ordinary data inside the workspace stay usable;
* nothing outside the workspace root is reachable: paths are checked after
  normalisation, so ``..``, absolute paths and symbolic links that leave the
  root are rejected (links are never followed out of the root).

Every solver, the prompt builder and the scratch copy go through this module.
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

# Top-level service files of a task: never data.
SERVICE_FILES = frozenset({"task.json", "answer.txt"})
_SKIP_DIRS = frozenset({"__pycache__", "__MACOSX"})


def _real(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def is_inside(root: str, path: str) -> bool:
    """True when ``path`` (after resolving links) is ``root`` or lies below it."""
    r, p = _real(root), _real(path)
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def _service_objects(root: str) -> List[str]:
    """Existing service files a path under ``root`` must never resolve to: those of
    the workspace itself and of the directories above it (the task directory when
    the workspace is ``<task>/workspace`` or a sub-directory such as ``kb/``)."""
    out = []
    d = os.path.abspath(root)
    for _ in range(3):
        for name in SERVICE_FILES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(p)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return out


def is_service_file(root: str, path: str, _targets: Optional[List[str]] = None) -> bool:
    """True if ``path`` is a top-level service file of ``root`` or an alias of a service file.

    The single predicate behind listing, preview, scratch copy and ``safe_join``.
    """
    if os.path.dirname(os.path.abspath(path)) == os.path.abspath(root) and \
            os.path.basename(path).lower() in SERVICE_FILES:
        return True
    real = _real(path)
    for target in (_service_objects(root) if _targets is None else _targets):
        if real == _real(target):
            return True
        try:
            if os.path.samefile(path, target):  # also catches hard links
                return True
        except OSError:
            pass
    return False


def safe_join(root: str, relative: str) -> Optional[str]:
    """``root/relative`` if it stays inside ``root`` and is not a service file, else None."""
    rel = (relative or "").strip()
    if not rel or "\x00" in rel or os.path.isabs(rel) or rel.startswith(("/", "\\")) or \
            (len(rel) > 1 and rel[1] == ":"):
        return None
    path = os.path.normpath(os.path.join(root, rel))
    if not is_inside(root, path) or _real(path) == _real(root):
        return None
    if is_service_file(root, path):
        return None
    return path


def safe_file(root: str, relative: str) -> Optional[str]:
    p = safe_join(root, relative)
    return p if p is not None and os.path.isfile(p) else None


def safe_dir(root: str, relative: str) -> Optional[str]:
    p = safe_join(root, relative)
    return p if p is not None and os.path.isdir(p) else None


def iter_files(root: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(relative path with '/', absolute path)`` of every allowed data file."""
    root = os.path.abspath(root)
    service = _service_objects(root)
    for d, dirs, files in os.walk(root, followlinks=False):
        top = os.path.abspath(d) == root
        dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS and not x.startswith(".")
                         and not os.path.islink(os.path.join(d, x)))
        for name in sorted(files):
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            if top and name.lower() in SERVICE_FILES:
                continue
            path = os.path.join(d, name)
            if not os.path.isfile(path) or not is_inside(root, path):
                continue  # broken link or a link that leaves the workspace
            if is_service_file(root, path, service):
                continue  # another name for task.json / answer.txt
            yield os.path.relpath(path, root).replace("\\", "/"), path


def list_files(root: str) -> List[Tuple[str, str]]:
    try:
        return list(iter_files(root))
    except OSError:
        return []


def top_level_names(root: str, suffixes: Tuple[str, ...]) -> List[str]:
    """Allowed files directly in ``root`` with one of the (lower-case) suffixes."""
    return [rel for rel, _p in list_files(root) if "/" not in rel and rel.lower().endswith(suffixes)]
