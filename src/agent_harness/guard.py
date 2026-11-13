"""Confinement wrapper for task code (unit tests of a task, model-written snippets).

The harness never runs task code directly; it starts

    python guard.py <jail> <private tmp> -c CODE | -m MODULE [args] | SCRIPT [args]

``jail`` is the scratch directory the child may read and write. Three layers
are applied inside the child before the task code starts:

1. resource limits (address space, file size, no core dumps);
2. Landlock (Linux >= 5.13, needs no privileges or capabilities, works under
   ``--cap-drop ALL`` / ``no-new-privileges`` / read-only root): the kernel then
   allows reading only the Python installation and system libraries, and
   reading/writing only the jail. The mounted task directory, other scratch
   copies, ``/proc`` (the environment of the harness) and ``/app`` are not
   reachable, also for grandchildren. With Landlock ABI >= 4 TCP connect/bind
   is denied as well;
3. a Python audit hook that refuses to open or list files outside the jail and
   the interpreter directories and refuses network connections made through
   Python's ``socket`` module. Without Landlock it additionally refuses to start
   processes and to load ``ctypes``, because those would escape an in-process hook.

Layer 3 alone is a guard against ordinary programs, not a security boundary
against deliberately hostile code; layer 2 is enforced by the kernel. The
active mode is printed by ``--probe`` and recorded in the run trace.

Known limit, measured with ``tests/net_isolation_probe.py`` on Landlock ABI 1
(kernel 5.15): the file system is confined by the kernel, the network is not.
The hook stops ``socket.connect`` in the first Python process only; a raw
``socket()/connect()`` through ctypes, a nested ``python`` started with
``subprocess`` and ``bash /dev/tcp`` did connect to a local listener. Only with
Landlock ABI >= 4 (kernel >= 6.7) TCP is denied by the kernel. The child has no
proxy key either way (scrubbed environment, no access to ``/proc`` or ``/app``).

This file must stay free of imports from the rest of the package: it is
executed as a plain script inside the child.
"""
import os
import sys

_FS_EXECUTE, _FS_WRITE_FILE, _FS_READ_FILE, _FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FS_REMOVE_DIR, _FS_REMOVE_FILE, _FS_MAKE_CHAR, _FS_MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
_FS_MAKE_REG, _FS_MAKE_SOCK, _FS_MAKE_FIFO, _FS_MAKE_BLOCK = 1 << 8, 1 << 9, 1 << 10, 1 << 11
_FS_MAKE_SYM, _FS_REFER, _FS_TRUNCATE, _FS_IOCTL_DEV = 1 << 12, 1 << 13, 1 << 14, 1 << 15
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1

_MEMORY_LIMIT = 4 * 1024 ** 3
_FILE_LIMIT = 128 * 1024 ** 2


def _limits() -> None:
    try:
        import resource
    except ImportError:  # not POSIX
        return
    for name, value in (("RLIMIT_AS", _MEMORY_LIMIT), ("RLIMIT_FSIZE", _FILE_LIMIT), ("RLIMIT_CORE", 0)):
        try:
            res = getattr(resource, name)
            _soft, hard = resource.getrlimit(res)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(res, (value, hard))
        except Exception:
            pass


def _system_read_roots():
    roots = {"/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/opt"}
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.realpath(sys.executable))):
        if p:
            roots.add(os.path.realpath(p))
    return sorted(r for r in roots if os.path.isdir(r))


def _landlock(rw_roots) -> int:
    """Apply a Landlock ruleset. Returns the ABI version in force, 0 if unavailable."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        abi = syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1))  # LANDLOCK_CREATE_RULESET_VERSION
        if abi < 1:
            return 0
        fs_all = (_FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_READ_DIR | _FS_REMOVE_DIR | _FS_REMOVE_FILE
                  | _FS_MAKE_CHAR | _FS_MAKE_DIR | _FS_MAKE_REG | _FS_MAKE_SOCK | _FS_MAKE_FIFO | _FS_MAKE_BLOCK
                  | _FS_MAKE_SYM)
        if abi >= 2:
            fs_all |= _FS_REFER
        if abi >= 3:
            fs_all |= _FS_TRUNCATE
        if abi >= 5:
            fs_all |= _FS_IOCTL_DEV
        file_bits = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | (fs_all & (_FS_TRUNCATE | _FS_IOCTL_DEV))

        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]

        class PathBeneath(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

        attr = RulesetAttr(fs_all, (_NET_BIND_TCP | _NET_CONNECT_TCP) if abi >= 4 else 0)
        size = ctypes.sizeof(RulesetAttr) if abi >= 4 else 8
        ruleset = syscall(444, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
        if ruleset < 0:
            return 0

        def allow(path: str, access: int) -> None:
            try:
                fd = os.open(path, getattr(os, "O_PATH", 0) | os.O_CLOEXEC)
            except OSError:
                return
            try:
                if not os.path.isdir(path):
                    access &= file_bits
                rule = PathBeneath(access & fs_all, fd)
                syscall(445, ctypes.c_int(ruleset), ctypes.c_int(1), ctypes.byref(rule), ctypes.c_uint32(0))
            finally:
                os.close(fd)

        read_only = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
        for root in _system_read_roots():
            allow(root, read_only)
        for dev in ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random"):
            allow(dev, _FS_READ_FILE | _FS_WRITE_FILE)
        for root in rw_roots:
            allow(root, fs_all)
        libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        if syscall(446, ctypes.c_int(ruleset), ctypes.c_uint32(0)) != 0:
            os.close(ruleset)
            return 0
        os.close(ruleset)
        return int(abi)
    except Exception:
        return 0


def _norm(path) -> str:
    p = os.fspath(path)
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    return os.path.normcase(os.path.realpath(p))


def _inside(path: str, roots) -> bool:
    for r in roots:
        if path == r or path.startswith(r.rstrip("\\/") + os.sep):
            return True
    return False


def _install_hook(rw_roots, landlocked: bool) -> None:
    rw = [_norm(r) for r in rw_roots]
    read_roots = [_norm(r) for r in _system_read_roots()] if sys.platform.startswith("linux") else []
    for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.realpath(sys.executable))}:
        if p:
            read_roots.append(_norm(p))
    devices = {os.path.normcase(os.devnull), "/dev/urandom", "/dev/random", "/dev/zero", "nul"}
    path_events = {"os.listdir", "os.scandir", "os.walk"}
    write_events = {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod", "os.chown", "os.link",
                    "os.symlink", "os.truncate", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copyfile",
                    "shutil.copytree", "shutil.chown", "shutil.make_archive", "shutil.unpack_archive"}
    spawn_events = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                    "os.forkpty", "os.startfile", "pty.spawn", "ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata",
                    "winreg.OpenKey", "_winapi.CreateProcess"}
    net_events = {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
                  "socket.gethostbyaddr", "socket.sendto", "urllib.Request", "http.client.connect",
                  "ftplib.connect", "smtplib.connect", "poplib.connect", "imaplib.open", "nntplib.connect",
                  "telnetlib.Telnet.open", "webbrowser.open"}

    def refuse(what: str):
        raise PermissionError("blocked by the harness sandbox: " + what)

    def check_path(path, writing: bool) -> None:
        if isinstance(path, int) or path is None:
            return
        try:
            p = _norm(path)
        except Exception:
            return
        if os.path.normcase(str(path)) in devices or p in devices:
            return
        if _inside(p, rw):
            return
        if not writing and _inside(p, read_roots):
            return
        refuse(("write to " if writing else "access to ") + str(path))

    def hook(event: str, args) -> None:
        if event == "open":
            mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
            writing = bool(isinstance(mode, str) and set(mode) & set("wax+")) or bool(
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            check_path(args[0], writing)
        elif event in path_events:
            check_path(args[0] if args else None, False)
        elif event in write_events:
            for a in args[:2]:
                if isinstance(a, (str, bytes, os.PathLike)):
                    check_path(a, True)
        elif event in net_events:
            refuse("network access")
        elif not landlocked and event in spawn_events:
            refuse(event)

    sys.addaudithook(hook)


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "--probe":
        jail = argv[2] if len(argv) > 2 else os.getcwd()
        abi = _landlock([jail])
        print("landlock_abi=%d" % abi)
        return 0
    if len(argv) < 4:
        sys.stderr.write("usage: guard.py JAIL TMP (-c CODE | -m MODULE | SCRIPT) [args]\n")
        return 2
    jail, tmp, rest = argv[1], argv[2], argv[3:]
    rw_roots = [jail] + ([tmp] if tmp and tmp != "-" else [])
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    sys.dont_write_bytecode = True

    _limits()
    landlocked = _landlock(rw_roots) > 0
    _install_hook(rw_roots, landlocked)

    import runpy

    sys.path.insert(0, os.getcwd())
    if rest[0] == "-c":
        sys.argv = ["-c"] + rest[2:]
        scope = {"__name__": "__main__", "__builtins__": __builtins__}
        exec(compile(rest[1], "<string>", "exec"), scope)
    elif rest[0] == "-m":
        sys.argv = [rest[1]] + rest[2:]
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        script = os.path.abspath(rest[0])
        sys.argv = [rest[0]] + rest[1:]
        sys.path[0] = os.path.dirname(script)
        runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
