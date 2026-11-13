#!/usr/bin/env bash
# Acceptance of a candidate image on a Linux host with Docker (development only).
#
#   CANDIDATE_SOURCE=/abs/path/to/exported/commit CANDIDATE_IMAGE=hack-candidate:<sha> bash tests/accept_docker.sh
#
# Works on fresh copies only: neither the source task-dir nor any other image
# or container is modified. Logs, traces and scores are kept in $AUDIT_RUN.
# No real proxy and no real key are used here.
set -uo pipefail
: "${CANDIDATE_SOURCE:?set absolute candidate source directory}"
: "${CANDIDATE_IMAGE:?set candidate image tag}"
# scripts mounted into the containers (default: the tests of the candidate itself)
TOOLS="${ACCEPT_TOOLS:-$CANDIDATE_SOURCE/tests}"
CPUS="${ACCEPT_CPUS:-4}"
MEM="${ACCEPT_MEMORY:-6g}"
AUDIT_RUN="$(mktemp -d /tmp/harness-final-verify.XXXXXX)"
export AUDIT_RUN
RUN_ID="$(basename "$AUDIT_RUN")"
NET="accept-net-$RUN_ID"
FAIL=0
note() { printf '\n===== %s =====\n' "$*"; }
verdict() { if [ "$1" = 0 ]; then echo "ACCEPT_OK   $2"; else echo "ACCEPT_FAIL $2"; FAIL=1; fi; }
cleanup() {
  for c in "verify-$RUN_ID" "mock-$RUN_ID"; do docker rm -f "$c" >/dev/null 2>&1 || true; done
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

SEC=(--read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp --cap-drop ALL --security-opt no-new-privileges
     --cpus "$CPUS" --memory "$MEM" --pids-limit 512)

fresh_tasks() {  # fresh_tasks <name> : pristine copy of the public examples, no answers
  local dst="$AUDIT_RUN/$1"
  cp -a "$CANDIDATE_SOURCE/task-dir" "$dst"
  find "$dst" -name answer.txt -type f -delete
  echo "$dst"
}
run_harness() {  # run_harness <tasks> <results> <log> [extra docker args...]
  local tasks="$1" results="$2" log="$3"; shift 3
  mkdir -p "$results" && chmod 777 "$results"
  timeout -k 15s 1h docker run --rm --name "verify-$RUN_ID" "$@" \
    -v "$tasks:/data/tasks:rw" -v "$results:/audit:rw" "${SEC[@]}" \
    --workdir /app --entrypoint python "$CANDIDATE_IMAGE" \
    /app/run_agent.py --tasks-dir /data/tasks --trace /audit/trace.jsonl > "$log" 2>&1
  local rc=$?
  docker rm -f "verify-$RUN_ID" >/dev/null 2>&1 || true
  return $rc
}

note "image"
docker images --format '{{.Repository}}:{{.Tag}} -> {{.Size}}' "$CANDIDATE_IMAGE"
echo "image id: $(docker image inspect "$CANDIDATE_IMAGE" --format '{{.Id}}')"
echo "host: $(nproc) cpu, $(free -g | awk '/Mem:/{print $2}') GB RAM, kernel $(uname -r), docker $(docker version --format '{{.Server.Version}}')"
echo "limits used here: --cpus $CPUS --memory $MEM (the rules say 8 cpu / 16 GB; the table of the rules says 4 cpu)"

note "image content: no tests, examples, mock, secrets; strict imports without network"
docker run --rm --network none "${SEC[@]}" --entrypoint sh "$CANDIDATE_IMAGE" -c '
  find /app -type f ! -name "*.pyc" | sort
  echo "dev files in the image: $(find / -xdev \( -name "task.json" -o -name "mock_llm_server.py" -o -name "score_local.py" -o -name "test_regressions.py" -o -name "*.tar.gz" \) 2>/dev/null | grep -v /usr/local/lib | wc -l)"
  cd /tmp && python -c "import sys; sys.path.insert(0, \"/app/src\"); import agent_harness.pipeline; from openjiuwen.core.foundation.llm import init_model; import openjiuwen; print(\"openjiuwen import ok, offline, read-only root\")"
  pip check 2>&1 | tail -1
  pip list 2>/dev/null | grep -i -E "^(openjiuwen|pytest) " ' > "$AUDIT_RUN/image_content.log" 2>&1
cat "$AUDIT_RUN/image_content.log"
grep -q "dev files in the image: 0" "$AUDIT_RUN/image_content.log"; verdict $? "no development files in the image"
grep -q "openjiuwen import ok" "$AUDIT_RUN/image_content.log"; verdict $? "openJiuwen imports offline on a read-only root"
grep -q "No broken requirements" "$AUDIT_RUN/image_content.log"; verdict $? "pip check"

note "A. free path, fresh writable copy, --network none, HARNESS_LLM=0"
T="$(fresh_tasks tasks_free)"; chmod -R a+rwX "$T"
run_harness "$T" "$AUDIT_RUN/results_free" "$AUDIT_RUN/run_free.log" --network none -e HARNESS_LLM=0 -e HARNESS_WORKERS=4
echo "container exit code: $?"
grep -E "isolation|SUMMARY|TOKENS|NOT WRITTEN|not writable|elapsed" "$AUDIT_RUN/run_free.log"
python3 "$CANDIDATE_SOURCE/tests/score_local.py" "$T" --no-run --trace "$AUDIT_RUN/results_free/trace.jsonl" \
  --original "$CANDIDATE_SOURCE/task-dir" | tee "$AUDIT_RUN/score_free.log"
grep -q "feasible: 8/8 correct, reward 20/20" "$AUDIT_RUN/score_free.log"; verdict $? "free path 8/8, 20/20, root answers, tests executed and unchanged"
[ "$(grep -c '"event": "llm_call"' "$AUDIT_RUN/results_free/trace.jsonl")" = 0 ]; verdict $? "free path made 0 LLM calls"
[ "$(find "$T" -maxdepth 2 -name answer.txt | wc -l)" = 8 ]; verdict $? "8 answer.txt files in the task directories"
[ "$(find "$T" -name '.tmp*' -o -name '__pycache__' | wc -l)" = 0 ]; verdict $? "no temp files or caches left in the tasks"

note "B. the command of the rules (network + env as there), unreachable proxy, fresh copy"
T="$(fresh_tasks tasks_rules)"; chmod -R a+rwX "$T"
docker network create --internal "$NET" >/dev/null
LLM_BASE_URL="http://mock-$RUN_ID:8000/v1" LLM_API_KEY="local-mock-key" \
run_harness "$T" "$AUDIT_RUN/results_rules" "$AUDIT_RUN/run_rules.log" --network "$NET" -e LLM_BASE_URL -e LLM_API_KEY
echo "container exit code: $?"
grep -E "SUMMARY|TOKENS|elapsed" "$AUDIT_RUN/run_rules.log"
grep -q "answered=8/8 reward=20/20" "$AUDIT_RUN/run_rules.log"; verdict $? "rules command: 8/8 with LLM env set and no proxy needed"

note "C. foreign owner, mode 775/755, no CAP_DAC_OVERRIDE: clean refusal"
T="$(fresh_tasks tasks_locked)"; chmod -R u=rwX,g=rwX,o=rX "$T"
stat -c 'task dirs: owner %u:%g mode %a' "$T/math_0000"
run_harness "$T" "$AUDIT_RUN/results_locked" "$AUDIT_RUN/run_locked.log" --network none -e HARNESS_LLM=0
RC=$?; echo "container exit code: $RC"
grep -E "NOT writable|NOT WRITTEN|SUMMARY" "$AUDIT_RUN/run_locked.log" | cut -c1-260
[ "$RC" = 0 ]; verdict $? "locked tasks: exit code 0"
[ "$(find "$T" -name answer.txt | wc -l)" = 0 ]; verdict $? "locked tasks: 0 answers, nothing faked"
grep -q "answered=0/8" "$AUDIT_RUN/run_locked.log"; verdict $? "locked tasks: no false SOLVED"
grep -q "NOT writable" "$AUDIT_RUN/run_locked.log"; verdict $? "locked tasks: diagnostic with owner / mode / uid"
diff -r -q "$CANDIDATE_SOURCE/task-dir" "$T" >/dev/null; verdict $? "locked tasks: task files untouched"

note "C2. one writable task among locked ones keeps its result"
T="$(fresh_tasks tasks_mixed)"; chmod -R u=rwX,g=rwX,o=rX "$T"; chmod -R a+rwX "$T/math_0004" "$T/coding_0003"
run_harness "$T" "$AUDIT_RUN/results_mixed" "$AUDIT_RUN/run_mixed.log" --network none -e HARNESS_LLM=0
grep -E "SUMMARY" "$AUDIT_RUN/run_mixed.log"
grep -q "answered=2/8" "$AUDIT_RUN/run_mixed.log" && [ -f "$T/math_0004/answer.txt" ] && [ -f "$T/coding_0003/answer.txt" ]
verdict $? "mixed permissions: the 2 writable tasks are answered, 6 reported as not written"

note "D. isolation of task code inside the container"
T="$(fresh_tasks tasks_iso)"; chmod -R a+rwX "$T"
docker run --rm --name "verify-$RUN_ID" --network none "${SEC[@]}" -e LLM_API_KEY=PROBE_KEY_MARKER \
  -v "$T:/data/tasks:rw" -v "$TOOLS:/t:ro" --workdir /tmp --entrypoint python \
  "$CANDIDATE_IMAGE" /t/docker_isolation_probe.py /data/tasks > "$AUDIT_RUN/isolation.log" 2>&1
RC=$?
cut -c1-330 "$AUDIT_RUN/isolation.log"
[ "$RC" = 0 ] && grep -q "ISOLATION_SUMMARY failed=none" "$AUDIT_RUN/isolation.log"; verdict $? "child code cannot reach task metadata, neighbours, the key or the network"

note "E. LLM tier through the installed openJiuwen against a local mock (canned answers: transport check only)"
docker run -d --name "mock-$RUN_ID" --network "$NET" -v "$TOOLS:/t:ro" \
  python:3.11.9-slim-bookworm python /t/mock_llm_server.py --host 0.0.0.0 --port 8000 >/dev/null
sleep 3
T="$(fresh_tasks tasks_mock)"
# two extra tasks for the generic fallback: one of them has no workspace/ and carries grading fields
mkdir -p "$T/fallback_0001/workspace" "$T/fallback_0002"
cat > "$T/fallback_0001/task.json" <<'JSON'
{"task_id": "fallback_0001", "task_type": "math", "instruction": "What is six times seven? Write ONLY the final answer to the file answer.txt in the current directory.", "reward": 1, "expected": "42", "check": "exact_int"}
JSON
cat > "$T/fallback_0002/task.json" <<'JSON'
{"task_id": "fallback_0002", "task_type": "math", "instruction": "What is forty plus two? Write ONLY the final answer to the file answer.txt in the current directory.", "reward": 1, "expected": "42", "check": "exact_int", "note": "GRADING_SENTINEL_NOTE"}
JSON
echo "some notes" > "$T/fallback_0002/notes.txt"
chmod -R a+rwX "$T"
LLM_BASE_URL="http://mock-$RUN_ID:8000/v1" LLM_API_KEY="local-mock-key" HARNESS_MUTATION_REPAIR=0 \
run_harness "$T" "$AUDIT_RUN/results_mock" "$AUDIT_RUN/run_mock.log" --network "$NET" \
  -e LLM_BASE_URL -e LLM_API_KEY -e HARNESS_MUTATION_REPAIR
echo "container exit code: $?"
grep -E "SUMMARY|TOKENS|fallback_000|coding_000" "$AUDIT_RUN/run_mock.log" | cut -c1-220
TR="$AUDIT_RUN/results_mock/trace.jsonl"
python3 - "$TR" <<'PY' | tee "$AUDIT_RUN/mock_trace_summary.log"
import json, sys
calls = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if '"llm_call"' in l]
for c in calls:
    print("MOCK_CALL", json.dumps({k: c.get(k) for k in ("task", "purpose", "model", "transport", "input_tokens",
                                                         "output_tokens", "reasoning_tokens", "cost",
                                                         "estimated_cost", "attempts", "usage_note")}))
blob = open(sys.argv[1], encoding="utf-8").read()
print("MOCK_TRANSPORTS", sorted({c["transport"] for c in calls}), "calls", len(calls))
prompts = " ".join(str(m.get("content")) for c in calls for m in (c.get("messages") or []))
marks = ("exact_int", "tests_pass", '"expected"', '"check"', "task.json", "GRADING_SENTINEL")
print("MOCK_SENTINEL_IN_TRACE", "GRADING_SENTINEL" in blob, "| grading fields in prompts:",
      any(m in prompts for m in marks))
PY
grep -q "MOCK_TRANSPORTS \['openjiuwen'\]" "$AUDIT_RUN/mock_trace_summary.log"; verdict $? "mock: all LLM calls went through openJiuwen"
grep -q "MOCK_SENTINEL_IN_TRACE False | grading fields in prompts: False" "$AUDIT_RUN/mock_trace_summary.log"; verdict $? "mock: no grading metadata in any prompt (task without workspace included)"
python3 "$CANDIDATE_SOURCE/tests/score_local.py" "$T" --no-run --trace "$TR" | tail -8 | tee "$AUDIT_RUN/score_mock.log"
grep -q "OK_LLM" "$AUDIT_RUN/score_mock.log" && [ "$(cat "$T/coding_0003/answer.txt")" = DONE ] && [ "$(cat "$T/coding_0007/answer.txt")" = DONE ]
verdict $? "mock: coding patches installed through the LLM tier, tests green, spend taken from the trace"
[ "$(cat "$T/fallback_0001/answer.txt" 2>/dev/null)" = 42 ] && [ "$(cat "$T/fallback_0002/answer.txt" 2>/dev/null)" = 42 ]
verdict $? "mock: generic fallback loop wrote its answers (with and without workspace/)"
docker logs "mock-$RUN_ID" 2>&1 | tail -2

note "F. proxy faults through the installed openJiuwen (local scripted server, no network)"
docker run --rm --name "verify-$RUN_ID" --network none "${SEC[@]}" -v "$TOOLS:/t:ro" \
  --workdir /tmp --entrypoint python "$CANDIDATE_IMAGE" /t/docker_llm_faults.py > "$AUDIT_RUN/faults.log" 2>&1
RC=$?
grep "FAULT_CHECK" "$AUDIT_RUN/faults.log" | cut -c1-300
[ "$RC" = 0 ] && grep -q "FAULT_CHECK_SUMMARY failed=none" "$AUDIT_RUN/faults.log"; verdict $? "429/500 recovery, Retry-After, auth stop, missing usage, time-out via real openJiuwen"

note "G. regression tests inside the image (python 3.11, uid 0 without capabilities)"
docker run --rm --name "verify-$RUN_ID" --network none "${SEC[@]}" -v "$CANDIDATE_SOURCE:/src:ro" \
  --workdir /tmp --entrypoint sh "$CANDIDATE_IMAGE" -c \
  'cp -r /src/tests /src/src /src/run_agent.py /tmp/ && cd /tmp && python -m unittest discover -s tests 2>&1 | tail -4' \
  > "$AUDIT_RUN/unittests_in_image.log" 2>&1
cat "$AUDIT_RUN/unittests_in_image.log"
grep -q "^OK" "$AUDIT_RUN/unittests_in_image.log"; verdict $? "unit tests and regressions pass inside the image"

note "result"
echo "artifacts: $AUDIT_RUN"
cleanup
echo "containers of this run still present after cleanup: $(docker ps -a --format '{{.Names}}' | grep -c "$RUN_ID")"
if [ "$FAIL" = 0 ]; then echo "ACCEPTANCE PASSED for $CANDIDATE_IMAGE"; else echo "ACCEPTANCE FAILED for $CANDIDATE_IMAGE"; fi
exit "$FAIL"
