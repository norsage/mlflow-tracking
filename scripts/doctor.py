#!/usr/bin/env python3
"""Check what a run needs to reach the MLflow server, without showing a secret.

    python3 doctor.py            # in the project, locally or in a remote task

Reads the project's mlflow.toml (the nearest one in the current directory or
above) and ~/.mlflow/credentials, then asks the server. Prints the file,
section and key names, the login name, modes and the server's answers;
never a password or a token. Exit 1 if anything a run needs is missing.
A local store (`sqlite:///<path>`) needs no login: only the database and the
packages MLflow opens it with (sqlalchemy, alembic) are checked.

Run it on every machine that logs: the laptop, and a remote task (with the
project's environment), which checks the credentials in that home directory
and the network from there. Needs mlflow-skinny, as find_runs.py.
"""

from __future__ import annotations

import argparse
import configparser
import importlib
import os
import re
import socket
import stat
import sys
from pathlib import Path

from _config import config_path, local_store, project_config

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")

ENV_SECRETS = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN")
KEYS = ("mlflow_tracking_username", "mlflow_tracking_password")
PLACEHOLDER_RE = re.compile(r"<[^<>]+>")
SECTION_TEMPLATE = """[mlflow]
mlflow_tracking_username = <your login on the server>
mlflow_tracking_password = <its password>"""


def placeholder(value: object) -> bool:
    """A template's `<...>` left unfilled."""
    return isinstance(value, str) and PLACEHOLDER_RE.search(value) is not None


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, what: str, text: str) -> None:
        if status == "FAIL":
            self.failed = True
        print(f"{status:<5} {what:<7} {text}")


def check_config(r: Report) -> dict:
    path = config_path()
    cfg = project_config()
    if os.environ.get("MLFLOW_TRACKING_URI"):
        r.line("ok", "server", f"MLFLOW_TRACKING_URI={os.environ['MLFLOW_TRACKING_URI']}")
        cfg = {**cfg, "tracking_uri": os.environ["MLFLOW_TRACKING_URI"]}
    if path is None:
        status = "ok" if cfg.get("tracking_uri") else "FAIL"
        r.line(status, "config", "no mlflow.toml in this directory or above")
        return cfg
    unfilled = [k for k, v in cfg.items() if placeholder(v)]
    if unfilled:
        r.line("FAIL", "config", f"{path}: fill in {', '.join(unfilled)}")
        return {k: v for k, v in cfg.items() if k not in unfilled}
    keys = ", ".join(f"{k}={v}" for k, v in cfg.items() if not isinstance(v, dict))
    r.line("ok" if cfg.get("tracking_uri") else "FAIL", "config", f"{path}: {keys or 'empty'}")
    return cfg


def check_env(r: Report) -> None:
    found = [k for k in ENV_SECRETS if os.environ.get(k)]
    if found:
        r.line(
            "WARN",
            "env",
            f"{', '.join(found)} set: they override ~/.mlflow/credentials and a remote "
            "task may show its environment; unset them",
        )
    else:
        r.line("ok", "env", "no MLflow credentials in the environment")


def check_credentials(r: Report) -> str | None:
    """Return the login name, if the file has one."""
    path = Path.home() / ".mlflow" / "credentials"
    if not path.is_file():
        r.line("FAIL", "creds", f"{path}: no file; create it (mode 600) with\n{SECTION_TEMPLATE}")
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    cp = configparser.ConfigParser()
    try:
        cp.read(path)
    except configparser.Error as e:
        r.line("FAIL", "creds", f"{path}: not an INI file ({type(e).__name__})")
        return None
    if "mlflow" not in cp:
        r.line("FAIL", "creds", f"{path}: no [mlflow] section; add\n{SECTION_TEMPLATE}")
        return None
    missing = [k for k in KEYS if not cp["mlflow"].get(k) or placeholder(cp["mlflow"].get(k))]
    user = None if KEYS[0] in missing else cp["mlflow"].get(KEYS[0])
    if missing:
        r.line("FAIL", "creds", f"{path} [mlflow]: fill in {', '.join(missing)}")
    else:
        r.line("ok", "creds", f"{path} [mlflow]: login {user}, password set")
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        r.line("FAIL", "creds", f"{path}: mode {mode:o}, others can read it; chmod 600 {path}")
    return user


def check_server(r: Report, uri: str) -> bool:
    """GET /health the way MLflow connects: requests, with certifi's CA bundle.

    Not urllib: it uses the system's CA store, which slim task images lack.
    """
    import requests

    verify: str | bool = os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH") or True
    if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() == "true":
        verify = False
    try:
        resp = requests.get(uri.rstrip("/") + "/health", timeout=15, verify=verify)
    except (requests.RequestException, ValueError) as e:
        r.line("FAIL", "server", f"{uri}: not reachable ({e})")
        return False
    # any answer means reachable; a login wall is checked next
    r.line("ok", "server", f"{uri}: /health {resp.status_code}")
    return True


def check_login(r: Report, uri: str, experiment: str | None, user: str | None) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    who = f"as {user}" if user else "without a login"
    try:
        client = MlflowClient()
        if experiment:
            exp = client.get_experiment_by_name(experiment)
            if exp is None:
                r.line("ok", "login", f"{who}; experiment {experiment!r} not created yet")
            else:
                r.line("ok", "login", f"{who}; experiment {experiment!r} id {exp.experiment_id}")
        else:
            client.search_experiments(max_results=1)
            r.line("ok", "login", f"{who}; no experiment in mlflow.toml")
    except Exception as e:  # noqa: BLE001 - every failure is a line of the report
        text = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        hint = ""
        if "401" in text or "Unauthorized" in text:
            hint = ": wrong login or password in ~/.mlflow/credentials"
        elif "403" in text or "Permission" in text:
            hint = f": {user or 'this login'} has no access; ask the server's administrator"
        r.line("FAIL", "login", f"{who}: {text}{hint}")


def check_endpoint(r: Report, endpoint: str | None) -> None:
    if not endpoint:
        r.line("ok", "s3", "no artifacts_endpoint in mlflow.toml: artifacts go through the server")
        return
    host, _, port = endpoint.removeprefix("https://").removeprefix("http://").partition(":")
    host = host.split("/")[0]
    try:
        socket.create_connection((host, int(port or 443)), timeout=5).close()
        r.line("ok", "s3", f"{host}:{port or 443} reachable (multipart upload of artifacts)")
    except (OSError, ValueError) as e:
        r.line("FAIL", "s3", f"{host}:{port or 443} not reachable ({e})")


def check_local_packages(r: Report) -> bool:
    """mlflow-skinny opens a sqlite store only with sqlalchemy and alembic installed;
    without them the run is not created and the computation goes on unlogged."""
    missing = []
    for name in ("sqlalchemy", "alembic"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        r.line(
            "FAIL",
            "store",
            f"{', '.join(missing)} not installed: mlflow-skinny cannot open a sqlite store "
            "and no run is created; uv add sqlalchemy alembic",
        )
        return False
    return True


def check_local(r: Report, db: Path, experiment: str | None) -> None:
    """A local store: its packages, and the database or the directory the first run
    creates it in."""
    packages = check_local_packages(r)
    if not db.exists():
        d = next(p for p in (db.parent, *db.parent.parents) if p.exists())
        if os.access(d, os.W_OK | os.X_OK):
            r.line("ok", "store", f"{db}: not created yet, the first run creates it")
        else:
            r.line("FAIL", "store", f"{db}: {d} is not writable")
        return
    if not os.access(db, os.W_OK) or not os.access(db.parent, os.W_OK | os.X_OK):
        r.line("FAIL", "store", f"{db}: not writable (the database and its directory)")
        return
    r.line("ok", "store", f"{db}: local store, no login")
    if packages:
        check_login(r, "sqlite:///" + str(db), experiment, None)


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    r = Report()
    cfg = check_config(r)
    check_env(r)
    uri = cfg.get("tracking_uri")
    db = local_store(uri)
    if db is not None:
        check_local(r, db, cfg.get("experiment"))
        return 1 if r.failed else 0
    user = check_credentials(r)
    if uri and check_server(r, uri):
        check_login(r, uri, cfg.get("experiment"), user)
    check_endpoint(r, cfg.get("artifacts_endpoint"))
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
