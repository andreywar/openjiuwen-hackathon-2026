#!/usr/bin/env python3
"""Can task code open a network connection from under the guard? (development only)

Unlike a connect to a closed port, this probe has evidence on both sides:

* it starts its own TCP listener on 127.0.0.1 and records what arrives;
* positive control: a child WITHOUT the guard must reach the listener, otherwise
  the result of the probe means nothing;
* then the same connection is attempted from under the guard in several ways -
  the Python ``socket`` module, a raw ``socket()/connect()`` through ctypes
  (invisible to Python audit hooks), a nested ``python -c`` started with
  ``subprocess`` (no hook inside it) and ``bash /dev/tcp``.

For every way the verdict is one of

    REACHED       the listener received the marker: the network is open this way
    BLOCKED       nothing arrived and the child reported a refusal (hook or kernel)
    UNSUPPORTED   the way does not exist on this platform (no bash, no libc, ...)
    NO_EVIDENCE   nothing arrived and the child gave no clear refusal

No external host, no key. Run it natively and inside the image:

    python tests/net_isolation_probe.py
    docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
        --cap-drop ALL --security-opt no-new-privileges -v "$PWD/tests:/t:ro" \
        --workdir /tmp --entrypoint python <image> /t/net_isolation_probe.py

(``--network none`` still has a loopback interface, which is all the probe needs.)
Exit code: 0 if the control worked (whatever the verdicts), 2 if the control failed.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/app/src" if os.path.isdir("/app/src") else os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from agent_harness import sandbox  # noqa: E402

received = []
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 0))
server.listen(16)
PORT = server.getsockname()[1]


def accept_loop() -> None:
    while True:
        try:
            conn, _addr = server.accept()
        except OSError:
            return
        try:
            conn.settimeout(3)
            received.append(conn.recv(64).decode("latin1", "replace"))
        except OSError:
            received.append("")
        finally:
            conn.close()


threading.Thread(target=accept_loop, daemon=True).start()

PY_SOCKET = ("import socket\ns = socket.socket(); s.settimeout(5); s.connect(('127.0.0.1', %d)); "
             "s.sendall(b'%s'); s.close(); print('SENT')")
CTYPES = r'''
import ctypes, os, socket, struct, sys
if os.name != "posix":
    print("UNSUPPORTED no libc"); sys.exit(0)
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.socket(2, 1, 0)                                  # AF_INET, SOCK_STREAM
if fd < 0:
    print("REFUSED socket() errno", ctypes.get_errno()); sys.exit(0)
addr = struct.pack("=H", 2) + struct.pack("!H", %d) + socket.inet_aton("127.0.0.1") + b"\0" * 8
if libc.connect(fd, addr, len(addr)) != 0:
    print("REFUSED connect() errno", ctypes.get_errno()); sys.exit(0)
libc.send(fd, b"%s", %d, 0)
libc.close(fd)
print("SENT")
'''
SUBPROCESS = ("import subprocess, sys\nr = subprocess.run([sys.executable, '-c', %r], capture_output=True, text=True, "
              "timeout=20)\nprint(r.stdout.strip() or 'CHILD_FAILED ' + r.stderr.strip()[-160:])")
BASH = ("import shutil, subprocess, sys\nif not shutil.which('bash'):\n    print('UNSUPPORTED no bash'); sys.exit(0)\n"
        "r = subprocess.run(['bash', '-c', 'echo -n %s > /dev/tcp/127.0.0.1/%d && echo SENT'], capture_output=True, "
        "text=True, timeout=20)\nprint(r.stdout.strip() or 'CHILD_FAILED ' + r.stderr.strip()[-160:])")


def ways():
    yield "python socket module", lambda m: PY_SOCKET % (PORT, m)
    yield "raw socket()/connect() through ctypes", lambda m: CTYPES % (PORT, m, len(m))
    yield "nested python via subprocess", lambda m: SUBPROCESS % (PY_SOCKET % (PORT, m))
    yield "bash /dev/tcp via subprocess", lambda m: BASH % (m, PORT)


def attempt(code: str, marker: str, guarded: bool, scratch: str):
    before = len(received)
    if guarded:
        r = sandbox.run_python(["-c", code], scratch, 30)
        out, err, rc = r.stdout, r.stderr, r.returncode
    else:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=scratch,
                           env=sandbox.child_env())
        out, err, rc = p.stdout, p.stderr, p.returncode
    time.sleep(0.3)
    arrived = any(marker in x for x in received[before:])
    text = (out + "\n" + err).strip()
    last = (text.splitlines() or [""])[-1][:200]
    if arrived:
        verdict = "REACHED"
    elif "UNSUPPORTED" in out:
        verdict = "UNSUPPORTED"
    elif "blocked by the harness sandbox" in text or "REFUSED" in out or "Permission denied" in text \
            or "PermissionError" in text or "CHILD_FAILED" in out or "Operation not permitted" in text:
        verdict = "BLOCKED"
    else:
        verdict = "NO_EVIDENCE"
    return verdict, rc, last


def main() -> int:
    mode = sandbox.isolation_mode()
    print("NET_PROBE_MODE " + json.dumps({"isolation": mode, "platform": sys.platform, "listener_port": PORT}),
          flush=True)
    base = tempfile.mkdtemp(prefix="ws_")
    scratch = os.path.join(base, "w")
    os.makedirs(scratch)
    control_ok = True
    results = {}
    try:
        for i, (name, make) in enumerate(ways()):
            marker_c, marker_g = "CONTROL%d" % i, "GUARDED%d" % i
            c_verdict, c_rc, c_last = attempt(make(marker_c), marker_c, False, scratch)
            g_verdict, g_rc, g_last = attempt(make(marker_g), marker_g, True, scratch)
            # a way is conclusive only if the same code reaches the listener without the guard
            conclusive = c_verdict == "REACHED"
            if i == 0 and not conclusive:
                control_ok = False
            results[name] = g_verdict if conclusive else ("UNSUPPORTED" if c_verdict == "UNSUPPORTED"
                                                          else "INCONCLUSIVE (control did not connect)")
            print("NET_PROBE " + json.dumps({"way": name, "without_guard": c_verdict, "under_guard": g_verdict,
                                             "conclusive": conclusive, "guarded_exit": g_rc,
                                             "guarded_says": g_last}), flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        server.close()
    open_ways = [k for k, v in results.items() if v == "REACHED"]
    print("NET_PROBE_SUMMARY " + json.dumps({"isolation": mode, "control_ok": control_ok,
                                             "network_reachable_via": open_ways, "verdicts": results}), flush=True)
    return 0 if control_ok else 2


if __name__ == "__main__":
    sys.exit(main())
