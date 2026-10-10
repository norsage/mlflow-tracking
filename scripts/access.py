#!/usr/bin/env python3
"""Give a person access to the project's MLflow experiment, or take it away.

    python3 access.py show <login>                     # their permission on the experiment
    python3 access.py grant <login>                    # EDIT: create runs and log to them
    python3 access.py grant <login> --permission MANAGE
    python3 access.py revoke <login>                   # back to the server's default

The experiment is the one of the project's mlflow.toml (or --experiment), the
server --tracking-uri, MLFLOW_TRACKING_URI or mlflow.toml. Run as a login with
MANAGE on the experiment: its creator has it, and whoever was granted it; an
administrator can do it for any experiment. The login to grant must already
exist on the server (an administrator creates it). Credentials come from
~/.mlflow/credentials; nothing secret is printed or taken as an argument.

Permissions: READ (see), USE, EDIT (create runs and log), MANAGE (also delete
the experiment and grant access). Works with both permission APIs: MLflow 3.10
(/api/2.0/mlflow/experiments/permissions/*) and 3.16
(/api/3.0/mlflow/users/permissions/*). Needs mlflow-skinny, as find_runs.py.
Exit 1 on a refusal or an error, with what to do.
"""

from __future__ import annotations

import argparse
import os
import sys

from _config import default_uri, project_config

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")

PERMISSIONS = ("READ", "USE", "EDIT", "MANAGE")
V3 = "/api/3.0/mlflow/users/permissions"
V2 = "/api/2.0/mlflow/experiments/permissions"


class Refused(Exception):
    """The server said no; the message says why and what to do."""


class Server:
    def __init__(self, uri: str) -> None:
        from mlflow.utils.credentials import get_default_host_creds

        self.creds = get_default_host_creds(uri)
        self.v3: bool | None = None  # found out on the first call

    def call(self, endpoint: str, method: str, **kwargs):
        from mlflow.utils.rest_utils import http_request

        resp = http_request(self.creds, endpoint, method, raise_on_status=False, **kwargs)
        assert resp is not None
        return resp

    def new_api(self, resp) -> bool:
        """A 404 that is not MLflow's JSON error: the route is not on this server."""
        if self.v3 is None:
            self.v3 = not (resp.status_code == 404 and not _json(resp))
        return self.v3


def _json(resp) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _missing(resp) -> str | None:
    """What a RESOURCE_DOES_NOT_EXIST is about: "permission" (none of the login's own) or "login"."""
    body = _json(resp)
    if body.get("error_code") != "RESOURCE_DOES_NOT_EXIST":
        return None
    return "permission" if "permission" in (body.get("message") or "").lower() else "login"


def _check(resp, login: str, experiment: str) -> dict:
    if resp.status_code == 200:
        return _json(resp)
    body = _json(resp)
    text = body.get("message") or resp.text.strip()[:200] or f"HTTP {resp.status_code}"
    if resp.status_code == 401:
        raise Refused("wrong login or password in ~/.mlflow/credentials (doctor.py checks them)")
    if resp.status_code == 403:
        raise Refused(
            f"no MANAGE on experiment {experiment!r} for this login: ask its creator, "
            "someone with MANAGE on it, or the server's administrator"
        )
    if _missing(resp) == "login":
        raise Refused(f"no login {login!r} on the server: the administrator creates it first")
    raise Refused(text)


def experiment_id(name: str) -> str:
    from mlflow.tracking import MlflowClient

    exp = MlflowClient().get_experiment_by_name(name)
    if exp is None:
        raise Refused(
            f"no experiment {name!r} on the server yet: the first run creates it, "
            "and its author gets MANAGE"
        )
    return exp.experiment_id


def show(s: Server, login: str, exp_id: str, exp: str) -> str:
    r = s.call(
        f"{V3}/get",
        "GET",
        params={"username": login, "resource_type": "experiment", "resource_id": exp_id},
    )
    if s.new_api(r):
        if r.status_code == 403:
            raise Refused(
                "this server shows a login's permission only to that login and to administrators; "
                "grant and revoke still work with MANAGE on the experiment"
            )
        return _check(r, login, exp)["permission"]
    r = s.call(f"{V2}/get", "GET", params={"experiment_id": exp_id, "username": login})
    if _missing(r) == "permission":
        return "the server's default (no permission of its own)"
    return _check(r, login, exp)["experiment_permission"]["permission"]


def grant(s: Server, login: str, exp_id: str, exp: str, permission: str) -> None:
    body = {
        "username": login,
        "resource_type": "experiment",
        "resource_id": exp_id,
        "permission": permission,
    }
    r = s.call(f"{V3}/grant", "POST", json=body)
    if s.new_api(r):
        if _json(r).get("error_code") == "RESOURCE_ALREADY_EXISTS":  # no update in this API
            _check(
                s.call(
                    f"{V3}/revoke",
                    "POST",
                    json={k: v for k, v in body.items() if k != "permission"},
                ),
                login,
                exp,
            )
            r = s.call(f"{V3}/grant", "POST", json=body)
        _check(r, login, exp)
        return
    body = {"experiment_id": exp_id, "username": login, "permission": permission}
    r = s.call(f"{V2}/create", "POST", json=body)
    if _json(r).get("error_code") == "RESOURCE_ALREADY_EXISTS":
        r = s.call(f"{V2}/update", "PATCH", json=body)
    _check(r, login, exp)


def revoke(s: Server, login: str, exp_id: str, exp: str) -> None:
    r = s.call(
        f"{V3}/revoke",
        "POST",
        json={"username": login, "resource_type": "experiment", "resource_id": exp_id},
    )
    if not s.new_api(r):
        r = s.call(f"{V2}/delete", "DELETE", json={"experiment_id": exp_id, "username": login})
    if _missing(r) == "permission":
        return  # nothing of its own to take away
    _check(r, login, exp)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("action", choices=("show", "grant", "revoke"))
    p.add_argument("login", help="the person's login on the MLflow server")
    p.add_argument(
        "--permission", choices=PERMISSIONS, default="EDIT", help="for grant; default EDIT"
    )
    p.add_argument("--experiment", help="default: experiment of mlflow.toml")
    p.add_argument("--tracking-uri", help="default: MLFLOW_TRACKING_URI, else mlflow.toml")
    a = p.parse_args()

    uri = a.tracking_uri or default_uri()
    exp = a.experiment or project_config().get("experiment")
    if not uri or not exp:
        print(
            "no server or experiment: run in the project (mlflow.toml) or pass "
            "--tracking-uri and --experiment",
            file=sys.stderr,
        )
        return 1
    import mlflow
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(uri)
    try:
        exp_id = experiment_id(exp)
        s = Server(uri)
        where = f"{a.login} on experiment {exp!r} (id {exp_id})"
        if a.action == "grant":
            grant(s, a.login, exp_id, exp, a.permission)
            print(f"{where}: {a.permission}")
        elif a.action == "revoke":
            revoke(s, a.login, exp_id, exp)
            print(f"{where}: the server's default (no permission of its own)")
        else:
            print(f"{where}: {show(s, a.login, exp_id, exp)}")
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    except MlflowException as e:  # finding the experiment: the client's own errors
        print(
            f"error: {str(e).splitlines()[0][:200]} (doctor.py checks the login)", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
