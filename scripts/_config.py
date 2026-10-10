"""The project's mlflow.toml, for the scripts of this skill.

The same lookup as `project_config` in templates/__pkg__/tracking.py: the nearest
mlflow.toml in the current directory or above, a relative `sqlite:///<path>`
made absolute relative to it. Committed, no secrets.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

CONFIG_FILE = "mlflow.toml"
SQLITE = "sqlite:///"


def config_path(start: str | os.PathLike | None = None) -> Path | None:
    d = Path(start or Path.cwd()).resolve()
    for p in (d, *d.parents):
        if (p / CONFIG_FILE).is_file():
            return p / CONFIG_FILE
    return None


def project_config(start: str | os.PathLike | None = None) -> dict[str, Any]:
    path = config_path(start)
    if path is None:
        return {}
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    uri = cfg.get("tracking_uri")
    if isinstance(uri, str) and uri.startswith(SQLITE) and not uri.startswith(SQLITE + "/"):
        cfg["tracking_uri"] = SQLITE + str(path.parent / uri.removeprefix(SQLITE))
    return cfg


def local_store(uri: str | None) -> Path | None:
    """The database file of a local store (`sqlite:///<path>`), else None: a server."""
    if isinstance(uri, str) and uri.startswith(SQLITE):
        return Path(uri.removeprefix(SQLITE)).resolve()
    return None


def default_uri() -> str | None:
    """MLFLOW_TRACKING_URI, else tracking_uri of mlflow.toml."""
    return os.environ.get("MLFLOW_TRACKING_URI") or project_config().get("tracking_uri")
