"""Writing ``answer.txt`` (and a repaired task module) atomically and verifiably.

Contract of a write: the new content goes to a temporary file in the target
directory, is flushed and fsynced, and only then replaces the target with
``os.replace``. A crash or an error before the replace leaves the previous file
intact and removes the temporary one. After the replace the file is read back
and compared with the exact bytes that were meant to be written - an old file
never counts as success.

What this module cannot do: create a file in a directory the process has no
write permission for. The container runs as uid 0 with ``--cap-drop ALL``, i.e.
*without* CAP_DAC_OVERRIDE / CAP_FOWNER, so ordinary Unix permissions apply to
it: a task directory must be writable for uid 0 through its owner, group or
"other" bits (for example owned by root, or ``chmod a+rwX``). Retrying or
``chmod`` from inside cannot change that, so nothing of the kind is attempted;
the failure is reported and the task is not counted as answered.

The required result is ``<task_dir>/answer.txt``. A second copy is put into
``workspace/`` (the "current directory" of the statement) when possible; it
never substitutes the required file.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

log = logging.getLogger("harness.answers")


def _fsync_dir(directory: str) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def describe_access(directory: str) -> str:
    """Owner / mode of a directory versus the ids of this process (for diagnostics)."""
    try:
        st = os.stat(directory)
        ids = "uid=%s gid=%s" % (os.getuid(), os.getgid()) if hasattr(os, "getuid") else "uid=n/a"
        return "directory owner=%s:%s mode=%o, process %s" % (st.st_uid, st.st_gid, st.st_mode & 0o7777, ids)
    except OSError as e:
        return "directory not accessible: %s" % e


def write_text_atomic(path: str, text: str) -> bool:
    """Atomically replace ``path`` with ``text`` and verify it. Never raises."""
    directory = os.path.dirname(os.path.abspath(path))
    data = text.encode("utf-8")
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".tmp_answer_", dir=directory)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            raise
        try:
            os.chmod(tmp, 0o666)  # our own new file: let the graders read / replace it
        except OSError:
            pass
        os.replace(tmp, path)
        tmp = None
        _fsync_dir(directory)
    except Exception as e:  # noqa: BLE001 - must never raise
        log.warning("cannot write %s: %s: %s (%s)", path, type(e).__name__, e, describe_access(directory))
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if _read_bytes(path) != data:
        log.warning("verification failed after writing %s", path)
        return False
    return True


# Backwards compatible name used by the solvers.
write_text_robust = write_text_atomic


def required_answer_path(task) -> str:
    return os.path.join(os.path.abspath(task.task_dir), "answer.txt")


def write_answer(task, answer: str) -> bool:
    """Persist the final answer. True only if ``<task_dir>/answer.txt`` holds exactly this answer."""
    text = str(answer).strip() + "\n"
    required = required_answer_path(task)
    ok = write_text_atomic(required, text)
    if not ok:
        log.error("[%s] required answer file could not be written: %s", task.task_id, required)
    extra_dir = os.path.abspath(task.workspace)
    if extra_dir != os.path.abspath(task.task_dir) and os.path.isdir(extra_dir):
        write_text_atomic(os.path.join(extra_dir, "answer.txt"), text)  # courtesy copy, not decisive
    return ok


def writable(directory: str) -> bool:
    """Can this process create files in ``directory``? (real test, then cleaned up)"""
    try:
        fd, tmp = tempfile.mkstemp(prefix=".tmp_probe_", dir=directory)
        os.close(fd)
        os.unlink(tmp)
        return True
    except Exception:  # noqa: BLE001
        return False
