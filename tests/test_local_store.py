"""A local store: mlflow.toml with sqlite:///<path>, no server, no credentials."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TEMPLATES = ROOT / "templates"

RUN = """
import mlflow
from __pkg__ import tracking
with tracking.start_run(params={"lr": 0.1}, task_id="t-local") as run:
    mlflow.log_metric("loss", 0.5, step=1)
    d = tracking.checkpoint_dir("ckpt", run)
    (d / "best.ckpt").write_bytes(b"weights")
    tracking.log_checkpoint(d / "best.ckpt")
    print("RUN", run.run_id)
"""


def env(home: Path) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("MLFLOW_", "RUNNER_"))}
    e.update(HOME=str(home), PYTHONPATH=str(TEMPLATES), MLFLOW_DISABLE_AGENT_HINT="1")
    return e


def run(args: list[str], cwd: Path, home: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env(home), capture_output=True, text=True, check=False
    )


def test_local_store_from_a_subdirectory(tmp_path):
    """The database and the artifacts sit beside mlflow.toml, wherever the run starts;
    find_runs, show_run and doctor work without a server and without a login."""
    proj, home = tmp_path / "proj", tmp_path / "home"
    sub = proj / "a" / "b"
    sub.mkdir(parents=True)
    home.mkdir()
    (proj / "mlflow.toml").write_text(
        'tracking_uri = "sqlite:///logs/mlflow.db"\nexperiment = "local"\n'
    )

    p = run([str(SCRIPTS / "doctor.py")], sub, home)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "not created yet" in p.stdout and "creds" not in p.stdout

    p = run(["-c", RUN], sub, home)
    assert p.returncode == 0, p.stderr
    run_id = next(s.split()[1] for s in p.stdout.splitlines() if s.startswith("RUN "))
    assert (proj / "logs" / "mlflow.db").is_file() and not (sub / "logs").exists()
    assert (proj / "logs" / "mlruns" / run_id / "artifacts" / "checkpoints" / "best.ckpt").is_file()
    assert oct((sub / "ckpt").stat().st_mode & 0o1777) != oct(0o1777)

    p = run([str(SCRIPTS / "show_run.py"), run_id, "--download", "dl/"], sub, home)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "ok" in p.stdout and (sub / "dl" / "checkpoints" / "best.ckpt").is_file()

    p = run([str(SCRIPTS / "find_runs.py"), "--task-id", "t-local"], sub, home)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"--run-id {run_id}" in p.stdout

    p = run([str(SCRIPTS / "doctor.py")], sub, home)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "local store, no login" in p.stdout and "experiment 'local' id" in p.stdout


def test_doctor_local_store_not_writable(tmp_path):
    proj, home = tmp_path / "proj", tmp_path / "home"
    (proj / "ro").mkdir(parents=True)
    home.mkdir()
    (proj / "mlflow.toml").write_text('tracking_uri = "sqlite:///ro/mlruns.db"\n')
    (proj / "ro").chmod(0o555)
    try:
        p = run([str(SCRIPTS / "doctor.py")], proj, home)
    finally:
        (proj / "ro").chmod(0o755)
    assert p.returncode == 1
    assert "not writable" in p.stdout


def skinny_version() -> str:
    """mlflow-skinny's version in uv.lock, the one the other tests run."""
    lock = (ROOT / "uv.lock").read_text()
    return re.search(r'name = "mlflow-skinny"\nversion = "([^"]+)"', lock).group(1)


def run_skinny(args: list[str], cwd: Path, home: Path, extra: tuple[str, ...] = ()):
    """In a fresh environment with mlflow-skinny only (plus `extra`): the dev group's full
    mlflow brings sqlalchemy and alembic, which a project with mlflow-skinny lacks."""
    e = {k: v for k, v in env(home).items() if not k.startswith(("VIRTUAL_ENV", "UV_"))}
    withs = [a for pkg in (f"mlflow-skinny=={skinny_version()}", *extra) for a in ("--with", pkg)]
    return subprocess.run(
        ["uv", "run", "--no-project", "--isolated", *withs, "python", *args],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_local_store_with_mlflow_skinny(tmp_path):
    """mlflow-skinny alone cannot open a sqlite store: doctor fails and says what to add;
    with sqlalchemy and alembic the run is created."""
    proj, home = tmp_path / "proj", tmp_path / "home"
    proj.mkdir()
    home.mkdir()
    (proj / "mlflow.toml").write_text(
        'tracking_uri = "sqlite:///logs/mlflow.db"\nexperiment = "local"\n'
    )

    p = run_skinny([str(SCRIPTS / "doctor.py")], proj, home)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "sqlalchemy, alembic not installed" in p.stdout, p.stdout
    assert "uv add sqlalchemy alembic" in p.stdout
    p = run_skinny(["-c", RUN], proj, home)
    assert "MLflow run not created" in p.stderr, p.stderr
    assert "RUN None" in p.stdout

    extra = ("sqlalchemy", "alembic")
    p = run_skinny([str(SCRIPTS / "doctor.py")], proj, home, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    p = run_skinny(["-c", RUN], proj, home, extra)
    assert p.returncode == 0, p.stderr
    run_id = next(s.split()[1] for s in p.stdout.splitlines() if s.startswith("RUN "))
    assert run_id != "None" and (proj / "logs" / "mlflow.db").is_file()
    p = run_skinny([str(SCRIPTS / "doctor.py")], proj, home, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "experiment 'local' id" in p.stdout
