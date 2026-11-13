# Harness image for the openJiuwen Challenge 2026.
#
#   docker build -t hack:latest .
#   docker run ... --entrypoint python hack:latest /app/run_agent.py --tasks-dir /data/tasks
#
# Everything is installed at build time; nothing is downloaded when the
# container runs (it only talks to the LLM proxy given in LLM_BASE_URL).

FROM python:3.11.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    TMPDIR=/tmp

WORKDIR /app

# openJiuwen with every transitive dependency pinned through the constraints
# file. The build fails if a pin cannot be installed: there is deliberately no
# fallback to unpinned versions.
COPY requirements.txt constraints.txt /app/
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt \
 && pip check

COPY run_agent.py /app/run_agent.py
COPY src/ /app/src/

# Byte-compile the harness (the root file system is read-only at run time) and
# check the imports the harness needs. A failed import fails the build.
RUN python -m compileall -q /app/src /app/run_agent.py \
 && cd /tmp \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import agent_harness.pipeline; from openjiuwen.core.foundation.llm import init_model; print('import check: ok')" \
 && rm -rf /tmp/* /root/.cache

ENTRYPOINT ["python"]
CMD ["/app/run_agent.py", "--tasks-dir", "/data/tasks"]
