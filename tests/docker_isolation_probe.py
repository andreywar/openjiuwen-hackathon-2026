#!/usr/bin/env python3
"""What can task code reach from inside the container? (development only)

Run inside the harness image with a task directory mounted, exactly with the
security flags of the competition command:

    docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
        --cap-drop ALL --security-opt no-new-privileges -e LLM_API_KEY=PROBE_KEY_MARKER \
        -v "$TASKS:/data/tasks:rw" -v "$PWD/tests:/t:ro" \
        --workdir /tmp --entrypoint python <image> /t/docker_isolation_probe.py /data/tasks

The script plays the harness: it makes a scratch copy of one task and runs
hostile snippets through ``agent_harness.sandbox.run_python`` - the only way
the harness executes task code. It uses its own markers and never prints real
secrets. Exit code 1 if any snippet reached something it must not reach.
"""
from __future__ import annotations

import json
import os
import sys

SRC = "/app/src" if os.path.isdir("/app/src") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from agent_harness import sandbox  # noqa: E402
from agent_harness.tasks import discover_tasks  # noqa: E402

tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/tasks"
tasks = discover_tasks(tasks_dir)
task = next(t for t in tasks if os.path.isdir(os.path.join(t.task_dir, "workspace")))
other = next(t for t in tasks if t.task_id != task.task_id)
meta = os.path.join(task.task_dir, "task.json")
neighbour = os.path.join(other.task_dir, "task.json")
marker = os.path.join(tasks_dir, "probe_marker_%d.txt" % os.getpid())
with open(marker, "w") as f:
    f.write("OWN_PROBE_MARKER")
other_scratch = sandbox.copy_workspace(other.workspace)
with open(os.path.join(other_scratch, "neighbour_marker.txt"), "w") as f:
    f.write("OWN_PROBE_MARKER")
scratch = sandbox.copy_workspace(task.workspace)
os.environ.setdefault("LLM_API_KEY", "PROBE_KEY_MARKER")

LEAKS = ("OWN_PROBE_MARKER", "PROBE_KEY_MARKER", '"expected"', "PWNED")
CASES = [
    # (name, python source, must_succeed)
    ("own scratch is readable and writable",
     "import os; open('probe_out.txt','w').write('x'); print(sorted(os.listdir('.'))[:3]); print('SCRATCH_OK')", True),
    ("read task.json of the task", "print(open(%r).read())" % meta, False),
    ("read task.json of a neighbour task", "print(open(%r).read())" % neighbour, False),
    ("read a marker in the mounted tasks dir", "print(open(%r).read())" % marker, False),
    ("list the mounted tasks dir", "import os; print(os.listdir(%r)); print('PWNED')" % tasks_dir, False),
    ("overwrite task.json", "open(%r,'w').write('{}'); print('PWNED')" % meta, False),
    ("create answer.txt of a neighbour task",
     "open(%r,'w').write('1'); print('PWNED')" % os.path.join(other.task_dir, "answer.txt"), False),
    ("read the scratch copy of another task",
     "print(open(%r).read())" % os.path.join(other_scratch, "neighbour_marker.txt"), False),
    ("read the harness environment through /proc",
     "import os; print(open('/proc/%d/environ','rb').read().decode('latin1'))" % os.getpid(), False),
    ("read the harness sources", "print(open('/app/run_agent.py').read()[:60]); print('PWNED')", False),
    ("key in the child's own environment",
     "import os; print(sorted(k for k in os.environ if 'LLM' in k or 'API' in k or 'HARNESS' in k)); "
     "print('ENV_CHECKED')", True),
    ("shell out to cat", "import subprocess; print(subprocess.run(['cat', %r], capture_output=True, text=True).stdout)"
     % meta, False),
    ("raw open(2) through ctypes (bypasses any Python-level hook)",
     "import ctypes, os\nlibc = ctypes.CDLL(None, use_errno=True)\nfd = libc.open(%r.encode(), 0)\n"
     "print('fd', fd)\nprint(os.read(fd, 400).decode() if fd >= 0 else 'DENIED_BY_KERNEL')" % meta, False),
    ("os.open with a relative path walking up",
     "import os; fd = os.open('../../../data/tasks/' + %r + '/task.json', os.O_RDONLY); print(os.read(fd, 400))"
     % os.path.basename(task.task_dir), False),
    ("symlink from the scratch to the tasks dir",
     "import os; os.symlink(%r, 'lnk'); print(open('lnk/' + %r + '/task.json').read())"
     % (tasks_dir, os.path.basename(task.task_dir)), False),
    ("TCP connection", "import socket; s = socket.socket(); s.settimeout(3); s.connect(('127.0.0.1', 9)); "
                       "print('PWNED')", False),
]

print("ISOLATION_MODE " + sandbox.isolation_mode(), flush=True)
bad = []
for name, code, must_succeed in CASES:
    r = sandbox.run_python(["-c", code], scratch, 20)
    leaked = [m for m in LEAKS if m in r.output]
    if must_succeed:
        ok = r.ok and not leaked
    else:
        ok = not leaked
    tail = (r.stderr.strip().splitlines() or [""])[-1][:110]
    print("ISOLATION_PROBE " + json.dumps(dict(case=name, ok=ok, child_exit=r.returncode,
                                               leaked=leaked, stdout=r.stdout.strip()[:70], error=tail)), flush=True)
    if not ok:
        bad.append(name)

intact = os.path.isfile(meta) and '"instruction"' in open(meta).read() and \
    not os.path.exists(os.path.join(other.task_dir, "answer.txt"))
print("ISOLATION_PROBE " + json.dumps(dict(case="task files intact after all attempts", ok=intact)))
if not intact:
    bad.append("task files modified")
os.unlink(marker)
sandbox.remove_scratch(scratch)
sandbox.remove_scratch(other_scratch)
print("ISOLATION_SUMMARY failed=%s" % (bad or "none"), flush=True)
sys.exit(1 if bad else 0)
