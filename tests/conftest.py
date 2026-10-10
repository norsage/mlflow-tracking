"""A real MLflow server (sqlite backend, served artifacts) for the whole session."""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tmp: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    mlflow_bin = Path(sys.executable).with_name("mlflow")
    proc = subprocess.Popen(
        [
            str(mlflow_bin),
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{tmp}/db.sqlite",
            "--artifacts-destination",
            str(tmp / "artifacts"),
            "--workers",
            "1",
        ],
        stdout=(tmp / "server.log").open("w"),
        stderr=subprocess.STDOUT,
        cwd=tmp,
        start_new_session=True,
    )
    uri = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(uri + "/health", timeout=1)
            return proc, uri
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError((tmp / "server.log").read_text())
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("mlflow server did not start")


def stop_server(proc: subprocess.Popen) -> None:
    """Kill the server and its workers (`mlflow server` forks uvicorn)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(10)


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlflow")
    proc, uri = start_server(tmp)
    yield uri
    stop_server(proc)


@pytest.fixture
def mlflow_env(server, monkeypatch, tmp_path):
    """Tracking URI set, RUNNER_* cleared, a fresh experiment, mlflow module restored after."""
    import mlflow

    from __pkg__ import tracking

    for v in (
        "RUNNER_GIT_SHA",
        "RUNNER_GIT_DIRTY",
        "RUNNER_TASK_ID",
        "RUNNER_EXP",
        "RUNNER_RESUME_RUN_ID",
        "TRACKING_RUN_ID",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "SLURM_PROCID",
        "JSM_NAMESPACE_RANK",
        "MLFLOW_RUN_ID",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server)
    mlflow.set_tracking_uri(server)
    monkeypatch.chdir(tmp_path)
    yield server
    tracking._restore()
    while mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(server)
