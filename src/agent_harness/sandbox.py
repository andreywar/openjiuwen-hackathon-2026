"""Running task code (tests, generated snippets) in child processes.

Task code never runs in the task directory and never directly: it is started
through ``guard.py`` on a scratch copy that holds only the allowed data files
of the task (see ``wsfiles``). The guard confines the child to that copy
(Landlock where the kernel provides it, a Python audit hook everywhere) and
sets resource limits. In addition the child gets a scrubbed environment (no
proxy key / URL), a private temporary directory, a wall-clock timeout after
which its whole process group is killed, and a cap on captured output.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from . import wsfiles

_SECRET_ENV = ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")
_GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard.py")
OUTPUT_LIMIT = 256 * 1024          # bytes kept per stream; the rest is read and dropped
COPY_LIMIT = 64 * 1024 * 1024      # bytes of task data copied into one scratch directory


def child_env(tmp_dir: Optional[str] = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in _SECRET_ENV and not k.upper().startswith(("LLM_", "OPENAI_", "HARNESS_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HOME", tempfile.gettempdir())
    if tmp_dir:
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = tmp_dir
        env["HOME"] = tmp_dir
    return env


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


def _jail_of(cwd: str) -> str:
    """Scratch copies live in ``<tmp>/ws_xxx/w``; the jail is the ``ws_xxx`` directory."""
    cwd = os.path.abspath(cwd)
    parent = os.path.dirname(cwd)
    return parent if os.path.basename(parent).startswith("ws_") else cwd


class _Collector(threading.Thread):
    """Drains a pipe, keeping at most ``limit`` bytes (head and tail)."""

    def __init__(self, stream, limit: int):
        super().__init__(daemon=True)
        self.stream, self.limit = stream, limit
        self.head = bytearray()
        self.tail = bytearray()
        self.truncated = False

    def run(self) -> None:
        half = self.limit // 2
        try:
            while True:
                chunk = self.stream.read(65536)
                if not chunk:
                    break
                room = half - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                if chunk:
                    self.tail += chunk
                    if len(self.tail) > half:
                        del self.tail[:len(self.tail) - half]
                        self.truncated = True
        except Exception:
            pass
        finally:
            try:
                self.stream.close()
            except Exception:
                pass

    def text(self) -> str:
        data = bytes(self.head) + (b"\n...[output truncated]...\n" if self.truncated else b"") + bytes(self.tail)
        return data.decode("utf-8", "replace").replace("\r\n", "\n")


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_python(args: List[str], cwd: str, timeout: float = 20.0, stdin: Optional[str] = None,
               deadline: Optional[float] = None) -> RunResult:
    """Run ``python <args>`` confined to the scratch directory ``cwd``; never raises.

    ``deadline`` (a ``time.monotonic()`` value) shortens the timeout so that a
    child never outlives the run.
    """
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0.2:
            return RunResult(-9, "", "deadline reached before the run", timed_out=True)
    jail = _jail_of(cwd)
    tmp_dir = os.path.join(jail, "tmp") if jail != os.path.abspath(cwd) else None
    if tmp_dir:
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError:
            tmp_dir = None
    cmd = [sys.executable, _GUARD, jail, tmp_dir or "-"] + list(args)
    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # so the whole process group can be killed
    else:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=child_env(tmp_dir),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    except Exception as e:  # noqa: BLE001
        return RunResult(127, "", "spawn failed: %s" % e)
    out, err = _Collector(proc.stdout, OUTPUT_LIMIT), _Collector(proc.stderr, OUTPUT_LIMIT)
    out.start()
    err.start()
    timed_out = False
    try:
        if stdin is not None:
            try:
                proc.stdin.write(stdin.encode("utf-8"))
                proc.stdin.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
    except BaseException:
        _kill(proc)
        raise
    finally:
        if timed_out or proc.poll() is None:
            _kill(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        elif os.name == "posix":
            try:  # stray grandchildren of a finished child
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        out.join(timeout=5)
        err.join(timeout=5)
    rc = proc.returncode if proc.returncode is not None else -9
    return RunResult(-9 if timed_out else rc, out.text(), err.text(), timed_out=timed_out,
                     truncated=out.truncated or err.truncated)


_PROBE: Optional[str] = None


def isolation_mode() -> str:
    """How child processes are confined on this machine (probed once)."""
    global _PROBE
    if _PROBE is None:
        mode = "audit-hook only (no Landlock on this kernel/platform)"
        try:
            d = tempfile.mkdtemp(prefix="ws_")
            try:
                r = subprocess.run([sys.executable, _GUARD, "--probe", d], cwd=d, env=child_env(), timeout=30,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                text = r.stdout.decode("utf-8", "replace")
                if "landlock_abi=" in text:
                    abi = int(text.split("landlock_abi=")[1].split()[0])
                    if abi > 0:
                        mode = "landlock abi %d + audit hook%s" % (
                            abi, "" if abi >= 4 else " (network is NOT restricted by the kernel)")
            finally:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        _PROBE = mode
    return _PROBE


def workspace_size(workspace: str) -> int:
    total = 0
    for _rel, path in wsfiles.list_files(workspace):
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def copy_workspace(workspace: str, max_bytes: int = COPY_LIMIT) -> Optional[str]:
    """Copy the allowed data files of a workspace to a scratch directory.

    Service files (``task.json``, ``answer.txt``) and anything reached through a
    link that leaves the workspace are not copied. Returns the new path or None.
    """
    dst = None
    try:
        files = wsfiles.list_files(workspace)
        total = 0
        for _rel, path in files:
            total += os.path.getsize(path)
            if total > max_bytes:
                return None
        dst = tempfile.mkdtemp(prefix="ws_")
        target = os.path.join(dst, "w")
        os.makedirs(target)
        for rel, path in files:
            out = os.path.join(target, *rel.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copyfile(path, out)  # follows links that stay inside the workspace
        return target
    except Exception:
        if dst:
            shutil.rmtree(dst, ignore_errors=True)
        return None


def remove_scratch(path: Optional[str]) -> None:
    """Delete a scratch copy made by ``copy_workspace`` (only inside the temp dir)."""
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    tmp_root = os.path.abspath(tempfile.gettempdir())
    if os.path.basename(parent).startswith("ws_") and os.path.dirname(parent) == tmp_root:
        shutil.rmtree(parent, ignore_errors=True)
