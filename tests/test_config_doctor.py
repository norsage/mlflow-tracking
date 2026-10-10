import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

from __pkg__ import tracking

ROOT = Path(__file__).resolve().parents[1]
DOCTOR = ROOT / "scripts" / "doctor.py"
TEMPLATES = ROOT / "templates"
PASSWORD = "s3cret-" + uuid.uuid4().hex[:6]


def write_toml(path: Path, uri: str, experiment: str, endpoint: str | None = None) -> None:
    text = f'tracking_uri = "{uri}"\nexperiment = "{experiment}"\n'
    if endpoint:
        text += f'artifacts_endpoint = "{endpoint}"\n'
    (path / "mlflow.toml").write_text(text)


def write_credentials(home: Path, user: str = "alice", mode: int = 0o600) -> Path:
    cred = home / ".mlflow" / "credentials"
    cred.parent.mkdir(parents=True, exist_ok=True)
    cred.write_text(
        f"[mlflow]\nmlflow_tracking_username = {user}\nmlflow_tracking_password = {PASSWORD}\n"
    )
    cred.chmod(mode)
    return cred


def test_mlflow_toml_gives_server_and_experiment(server, tmp_path):
    """A process with no MLflow settings at all finds mlflow.toml in a parent directory."""
    name = f"cfg-{uuid.uuid4().hex[:6]}"
    write_toml(tmp_path, server, name)
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    env["PYTHONPATH"] = str(TEMPLATES)
    env["HOME"] = str(tmp_path)
    code = "from __pkg__ import tracking\nwith tracking.start_run(sha='abc') as r:\n    print('RUN', r.run_id)"
    p = subprocess.run(
        [sys.executable, "-c", code], cwd=sub, env=env, capture_output=True, text=True, check=True
    )
    mlflow.set_tracking_uri(server)
    run_id = next(s.split()[1] for s in p.stdout.splitlines() if s.startswith("RUN "))
    r = MlflowClient().get_run(run_id)
    assert MlflowClient().get_experiment(r.info.experiment_id).name == name


def test_environment_wins_over_mlflow_toml(mlflow_env, tmp_path):
    name = f"cfg-{uuid.uuid4().hex[:6]}"
    write_toml(tmp_path, "http://127.0.0.1:1", name)
    with tracking.start_run() as run:
        pass
    assert run.active
    r = MlflowClient().get_run(run.run_id)
    assert MlflowClient().get_experiment(r.info.experiment_id).name == name


def test_author_is_the_server_login(mlflow_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_credentials(tmp_path, "alice")
    with tracking.start_run(experiment=f"u-{uuid.uuid4().hex[:6]}") as run:
        pass
    r = MlflowClient().get_run(run.run_id)
    assert r.data.tags["mlflow.user"] == "alice"
    assert r.info.user_id == "alice"


def doctor(cwd: Path, home: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_TRACKING")}
    env["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, str(DOCTOR)], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def test_doctor_all_good_never_prints_the_password(server, tmp_path):
    port = urlparse(server).port
    write_toml(tmp_path, server, f"d-{uuid.uuid4().hex[:6]}", f"127.0.0.1:{port}")
    write_credentials(tmp_path, "alice")
    p = doctor(tmp_path, tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "login alice, password set" in p.stdout
    assert "not created yet" in p.stdout
    assert PASSWORD not in p.stdout + p.stderr


def test_doctor_reports_missing_and_open_credentials(server, tmp_path):
    write_toml(tmp_path, server, "x")
    p = doctor(tmp_path, tmp_path)
    assert p.returncode == 1
    assert "no file" in p.stdout and "[mlflow]" in p.stdout

    write_credentials(tmp_path, mode=0o644)
    p = doctor(tmp_path, tmp_path)
    assert p.returncode == 1
    assert "others can read it" in p.stdout
    assert PASSWORD not in p.stdout + p.stderr


def test_doctor_without_config(tmp_path):
    p = doctor(tmp_path, tmp_path)
    assert p.returncode == 1
    assert "no mlflow.toml" in p.stdout


def test_doctor_rejects_template_placeholders(tmp_path):
    (tmp_path / "mlflow.toml").write_text(
        'tracking_uri = "<https://mlflow-host>"\nexperiment = "x"\n'
        'artifacts_endpoint = "<s3-host>:443"\n'
    )
    cred = tmp_path / ".mlflow" / "credentials"
    cred.parent.mkdir()
    cred.write_text(
        "[mlflow]\nmlflow_tracking_username = <login>\nmlflow_tracking_password = <pw>\n"
    )
    cred.chmod(0o600)
    p = doctor(tmp_path, tmp_path)
    assert p.returncode == 1
    assert "fill in tracking_uri, artifacts_endpoint" in p.stdout
    assert "fill in mlflow_tracking_username, mlflow_tracking_password" in p.stdout
    assert "not reachable" not in p.stdout  # no request to a placeholder
