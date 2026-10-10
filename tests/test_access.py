"""access.py against basic-auth servers: the dev MLflow (users/permissions API) and
3.10, a version still in use (experiments/permissions API)."""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import requests
from conftest import free_port, stop_server

ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT / "scripts" / "access.py"
PW = {
    "admin": "admin-password-0123",
    "owner": "owner-password-0123",
    "alice": "alice-password-0123",
    "bob": "bob-password-0123",
}
OLD = "3.10.0"


@pytest.fixture(scope="module", params=["dev", OLD])
def auth_server(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"auth-{request.param}")
    ini = tmp / "basic_auth.ini"
    ini.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = sqlite:///{tmp}/auth.db\n"
        "admin_username = admin\n"
        f"admin_password = {PW['admin']}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
    )
    mlflow = (
        [str(Path(sys.executable).with_name("mlflow"))]
        if request.param == "dev"
        else ["uvx", "--from", f"mlflow[auth]=={OLD}", "mlflow"]
    )
    port = free_port()
    proc = subprocess.Popen(
        [
            *mlflow,
            "server",
            "--app-name",
            "basic-auth",
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
        env={
            **os.environ,
            "MLFLOW_AUTH_CONFIG_PATH": str(ini),
            "MLFLOW_FLASK_SERVER_SECRET_KEY": "test-secret",
        },
    )
    uri = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            urllib.request.urlopen(uri + "/health", timeout=1)
            break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError((tmp / "server.log").read_text())
            time.sleep(0.5)
    else:
        stop_server(proc)
        raise RuntimeError("mlflow server did not start")

    admin = ("admin", PW["admin"])
    for user in ("owner", "alice", "bob"):
        r = requests.post(
            f"{uri}/api/2.0/mlflow/users/create",
            auth=admin,
            json={"username": user, "password": PW[user]},
        )
        assert r.status_code == 200, r.text
    r = requests.post(
        f"{uri}/api/2.0/mlflow/experiments/create",
        auth=("owner", PW["owner"]),
        json={"name": "proj"},
    )
    assert r.status_code == 200, r.text
    yield uri, r.json()["experiment_id"], request.param
    stop_server(proc)


def access(uri, as_user, *args):
    env = {
        **os.environ,
        "MLFLOW_TRACKING_USERNAME": as_user,
        "MLFLOW_TRACKING_PASSWORD": PW[as_user],
    }
    env.pop("MLFLOW_TRACKING_URI", None)
    r = subprocess.run(
        [sys.executable, str(ACCESS), *args, "--tracking-uri", uri, "--experiment", "proj"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert not any(pw in r.stdout + r.stderr for pw in PW.values())
    return r


def can_log(uri, exp_id, user):
    r = requests.post(
        f"{uri}/api/2.0/mlflow/runs/create", auth=(user, PW[user]), json={"experiment_id": exp_id}
    )
    return r.status_code == 200


def test_grant_update_revoke(auth_server):
    uri, exp_id, version = auth_server
    assert not can_log(uri, exp_id, "alice")

    r = access(uri, "owner", "grant", "alice")
    assert r.returncode == 0, r.stderr
    assert f"alice on experiment 'proj' (id {exp_id}): EDIT" in r.stdout
    assert can_log(uri, exp_id, "alice")
    assert not can_log(uri, exp_id, "bob")

    r = access(uri, "owner", "grant", "alice", "--permission", "MANAGE")
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith("MANAGE")
    # MANAGE passes the right on: alice now grants bob
    r = access(uri, "alice", "grant", "bob", "--permission", "READ")
    assert r.returncode == 0, r.stderr

    for _ in range(2):  # the second time there is nothing to take away
        r = access(uri, "owner", "revoke", "alice")
        assert r.returncode == 0, r.stderr
    assert not can_log(uri, exp_id, "alice")
    r = access(uri, "owner", "show", "alice")
    if version == "dev":  # users/permissions/get: only the login itself and administrators
        assert r.returncode == 1
        assert "only to that login and to administrators" in r.stderr
        r = access(uri, "alice", "show", "alice")
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith(("READ", "the server's default (no permission of its own)"))


def test_refusals(auth_server):
    uri, _, version = auth_server
    r = access(uri, "bob", "grant", "bob")
    assert r.returncode == 1
    assert "no MANAGE on experiment 'proj'" in r.stderr

    r = access(uri, "owner", "grant", "nobody")
    assert r.returncode == 1
    assert "no login 'nobody' on the server" in r.stderr

    if version != "dev":  # on dev, showing another login is refused before the login is looked up
        r = access(uri, "owner", "show", "nobody")
        assert r.returncode == 1
        assert "no login 'nobody' on the server" in r.stderr


def test_experiment_not_created_yet(auth_server):
    uri, _, _ = auth_server
    env = {
        **os.environ,
        "MLFLOW_TRACKING_USERNAME": "owner",
        "MLFLOW_TRACKING_PASSWORD": PW["owner"],
    }
    r = subprocess.run(
        [
            sys.executable,
            str(ACCESS),
            "grant",
            "alice",
            "--tracking-uri",
            uri,
            "--experiment",
            "later",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert r.returncode == 1
    assert "no experiment 'later' on the server yet" in r.stderr
