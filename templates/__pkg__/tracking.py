"""Open an MLflow run with reproducibility tags, a rank guard and network resilience.

Copied into the project from the mlflow-tracking skill; the project owns this
copy and decides when to update it. Needs `mlflow-skinny` (or `mlflow`).

    from __pkg__ import tracking
    with tracking.start_run(experiment="my-project", params=cfg) as run:
        mlflow.log_metric("loss", 0.1, step=1)   # plain MLflow API, active run

The server and the experiment: arguments, else MLFLOW_TRACKING_URI and
MLFLOW_EXPERIMENT_NAME, else the project's mlflow.toml (`project_config`).
The server may be a local store, `sqlite:///<path>` relative to mlflow.toml.
Credentials (a server with a login): ~/.mlflow/credentials, read by MLflow itself.

Tags on every run:
    git.sha, git.dirty   argument, RUNNER_GIT_SHA (+ RUNNER_GIT_DIRTY), else `git`;
                         dirty counts untracked files too, except *.md
    git.dirty.paths      the uncommitted paths, when dirty from `git`
    runner.task_id       argument or RUNNER_TASK_ID; not set for a local run
    exp                  argument or RUNNER_EXP: the research-log experiment id
    sweep.role           parent, child or single
    init_from.run_id     the run whose checkpoints this one starts from
    mlflow.user          the server login, else the OS user (MLflow's run author)

RUNNER_* are this skill's variables; a transport sets them from its own.

A run is resumed only from the commit it was started with. Another commit
gets a new run that starts from the old one's checkpoints (init_from).

Only rank 0 opens a run. Other ranks and a process whose run could not be
created get an inactive handle, and MLflow's logging functions become no-ops
there. A server that goes away mid-run costs metrics, not the computation.
"""

from __future__ import annotations

import functools
import getpass
import hashlib
import logging
import os
import re
import stat
import subprocess
import time
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self

import mlflow
import mlflow.config
import mlflow.tracking.fluent
from mlflow.entities import Param
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

ENV_SHA = "RUNNER_GIT_SHA"
ENV_DIRTY = "RUNNER_GIT_DIRTY"
ENV_TASK_ID = "RUNNER_TASK_ID"
ENV_EXP = "RUNNER_EXP"
ENV_RESUME = "RUNNER_RESUME_RUN_ID"
ENV_INIT_FROM = "RUNNER_INIT_FROM_RUN_ID"
# Set by rank 0 for the processes it spawns (Lightning DDP): they learn the
# run id without opening the run. Not MLFLOW_RUN_ID, which MLflow acts on.
ENV_RUN_ID = "TRACKING_RUN_ID"
# The same for init_from: the run whose checkpoints the new run starts from.
ENV_RUN_INIT = "TRACKING_INIT_FROM"

TAG_SHA = "git.sha"
TAG_DIRTY = "git.dirty"
TAG_DIRTY_PATHS = "git.dirty.paths"
TAG_TASK_ID = "runner.task_id"
TAG_EXP = "exp"
TAG_ROLE = "sweep.role"
TAG_PARENT = "mlflow.parentRunId"
TAG_INIT_FROM = "init_from.run_id"
TAG_USER = "mlflow.user"
ROLES = ("parent", "child", "single")
CONFIG_FILE = "mlflow.toml"
SQLITE = "sqlite:///"

# Fluent functions that talk to the server synchronously even with async
# logging on. They are wrapped: a failure is a warning, not an exception.
_GUARDED = (
    "end_run",
    "set_tag",
    "set_tags",
    "log_param",
    "log_params",
    "log_metric",
    "log_metrics",
    "log_artifact",
    "log_artifacts",
    "log_dict",
    "log_text",
    "log_figure",
    "log_image",
    "log_table",
    "log_input",
    "set_terminated",
)
# Made no-ops where this process must not write (rank != 0, no run).
_SILENCED = _GUARDED + ("start_run", "autolog")


def rank_zero() -> bool:
    """The same test as Lightning's rank_zero_only at import: the first of RANK,
    LOCAL_RANK, SLURM_PROCID, JSM_NAMESPACE_RANK that is set; plus NODE_RANK,
    since the main process of node 1 also has LOCAL_RANK=0. Lightning's
    subprocess launcher sets LOCAL_RANK before it re-runs the script, so the
    test holds from the first line of every rank."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        if os.environ.get(key):
            return int(os.environ[key]) == 0 and os.environ.get("NODE_RANK", "0") in ("", "0")
    return os.environ.get("NODE_RANK", "0") in ("", "0")


class Run:
    """Handle returned by start_run. `active` is False on rank != 0 and when
    the run could not be created; `run_id` may still be known there.
    `resumed`: an unfinished run was reopened. `init_from`: the run whose
    checkpoints this new run starts from (init_checkpoint_dir)."""

    def __init__(
        self, run_id: str | None, active: bool, resumed: bool = False, init_from: str | None = None
    ):
        self.run_id = run_id
        self.active = active
        self.resumed = resumed
        self.init_from = init_from

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            end_run("KILLED")
        else:
            end_run("FINISHED" if exc_type is None else "FAILED")

    def __repr__(self) -> str:
        return (
            f"Run(run_id={self.run_id!r}, active={self.active}, resumed={self.resumed}, "
            f"init_from={self.init_from!r})"
        )


def project_config(start: str | os.PathLike | None = None) -> dict[str, Any]:
    """The project's mlflow.toml: the nearest one in `start` (default: cwd) or above.

    Committed, no secrets: tracking_uri, experiment, and optionally
    shared_checkpoints (`checkpoint_dir`) and artifacts_endpoint (the S3
    endpoint the server's multipart upload sends parts to; only doctor.py
    reads it). A relative `sqlite:///<path>` comes back absolute, relative
    to mlflow.toml: the same database from any directory. {} if there is none.
    """
    d = Path(start or Path.cwd()).resolve()
    for p in (d, *d.parents):
        f = p / CONFIG_FILE
        if f.is_file():
            with f.open("rb") as fh:
                cfg = tomllib.load(fh)
            uri = cfg.get("tracking_uri")
            if isinstance(uri, str) and uri.startswith(SQLITE) and not uri.startswith(SQLITE + "/"):
                cfg["tracking_uri"] = SQLITE + str(p / uri.removeprefix(SQLITE))
            return cfg
    return {}


def start_run(
    *,
    experiment: str | None = None,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    sha: str | None = None,
    task_id: str | None = None,
    exp: str | None = None,
    role: str = "single",
    parent_run_id: str | None = None,
    resume_run_id: str | None = None,
    init_from: str | None = None,
    resume_match: Mapping[str, Any] | None = None,
    tracking_uri: str | None = None,
    activate: bool = True,
) -> Run:
    """Open (or resume) a run and make it active.

    experiment, tracking_uri: None takes MLflow's own defaults
      (MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI), else mlflow.toml.
    params: nested mappings are flattened with dots.
    resume_run_id: reopen this run. Default: RUNNER_RESUME_RUN_ID for a
      top-level run. Without it, a run with the same runner.task_id, role
      and `resume_match` tags that did not finish is reopened. A run is
      reopened only if its git.sha is this commit; otherwise a new run
      starts from it (init_from), and a sweep's children follow.
    init_from: start a new run from this run's checkpoints, tagged
      init_from.run_id. Default: RUNNER_INIT_FROM_RUN_ID for a top-level run.
    activate: False creates the run without making it active in this
      process (a sweep parent whose children run elsewhere).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if not rank_zero():
        _silence("rank != 0")
        return Run(
            os.environ.get(ENV_RUN_ID) or None,
            active=False,
            init_from=os.environ.get(ENV_RUN_INIT) or None,
        )

    _restore()
    os.environ.pop(ENV_RUN_ID, None)  # left by the previous job of a sweep
    os.environ.pop(ENV_RUN_INIT, None)
    _check_secrets()
    cfg = project_config()
    if not tracking_uri and not os.environ.get("MLFLOW_TRACKING_URI"):
        tracking_uri = cfg.get("tracking_uri")
    if experiment is None and not os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        experiment = cfg.get("experiment")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.config.enable_async_logging(True)
    _guard()

    all_tags = {str(k): str(v) for k, v in (tags or {}).items()}
    all_tags.update(_repro_tags(sha, task_id, exp))
    all_tags[TAG_ROLE] = role
    all_tags.setdefault(TAG_USER, _user())
    if parent_run_id:
        all_tags[TAG_PARENT] = parent_run_id
    if run_name:
        all_tags["mlflow.runName"] = run_name
    if resume_run_id is None and role != "child":
        resume_run_id = os.environ.get(ENV_RESUME) or None
    if init_from is None and role != "child":
        init_from = os.environ.get(ENV_INIT_FROM) or None

    try:
        client = MlflowClient()
        exp_id = _experiment_id(client, experiment)
        run_id = None
        if not init_from:
            run_id = resume_run_id or _find_unfinished(client, exp_id, all_tags, role, resume_match)
        if run_id and not _same_commit(client, run_id, all_tags.get(TAG_SHA)):
            run_id, init_from = None, run_id
        if not run_id and not init_from and role == "child":
            init_from = _child_of_init_parent(client, exp_id, all_tags, resume_match)
        if init_from:
            all_tags[TAG_INIT_FROM] = init_from
        resumed = bool(run_id)
        if run_id:
            log.info("resuming MLflow run %s", run_id)
            for k, v in all_tags.items():
                if k not in ("mlflow.runName", TAG_USER):  # the author stays
                    client.set_tag(run_id, k, v)
            client.update_run(run_id, status="RUNNING")
        else:
            run_id = client.create_run(exp_id, tags=all_tags, run_name=run_name).info.run_id
        if activate:
            mlflow.start_run(run_id=run_id)
    except Exception as e:  # noqa: BLE001 - any failure here must not stop the computation
        log.warning("MLflow run not created, the computation goes on without it: %s", e)
        _silence("no run")
        return Run(None, active=False)

    if activate:
        os.environ[ENV_RUN_ID] = run_id
        if init_from:
            os.environ[ENV_RUN_INIT] = init_from
    if params:
        log_params(params, run_id=run_id)
    return Run(run_id, active=True, resumed=resumed, init_from=init_from)


def end_run(status: str = "FINISHED") -> None:
    """End the active run, if any. Never raises."""
    if not rank_zero():
        return
    os.environ.pop(ENV_RUN_ID, None)
    os.environ.pop(ENV_RUN_INIT, None)
    if mlflow.active_run() is None:
        return
    try:
        mlflow.flush_async_logging()
    except Exception as e:  # noqa: BLE001
        log.warning("MLflow: pending metrics not sent: %s", e)
    mlflow.end_run(status)  # guarded


def terminate(run_id: str, status: str = "FINISHED") -> None:
    """End a run that is not active in this process (a sweep parent). Never raises."""
    try:
        MlflowClient().set_terminated(run_id, status)
    except Exception as e:  # noqa: BLE001
        log.warning("MLflow: run %s not terminated: %s", run_id, e)


def log_params(params: Mapping[str, Any], run_id: str | None = None) -> None:
    """Log a nested mapping as flat params; a value that cannot be logged is skipped."""
    flat = flatten(params)
    active = mlflow.active_run()
    run_id = run_id or (active.info.run_id if active else None)
    if not run_id or not flat:
        return
    client = MlflowClient()
    existing = {}
    try:
        existing = client.get_run(run_id).data.params
    except Exception as e:  # noqa: BLE001 - unknown params: log them all, the server decides
        log.debug("MLflow: params of %s not read: %s", run_id, e)
    items = [(k, v) for k, v in flat.items() if k not in existing]
    changed = [k for k, v in flat.items() if k in existing and existing[k] != v]
    if changed:
        log.warning(
            "MLflow params are immutable; kept the old value of %s", ", ".join(changed[:10])
        )
    for i in range(0, len(items), 100):
        batch = [Param(k, v) for k, v in items[i : i + 100]]
        try:
            client.log_batch(run_id, params=batch, synchronous=False)
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow params not logged: %s", e)


def flatten(d: Mapping[str, Any], prefix: str = "", max_len: int = 6000) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, key + ".", max_len))
        else:
            s = str(v)
            out[key] = s if len(s) <= max_len else s[: max_len - 3] + "..."
    return out


# --- checkpoints ---------------------------------------------------------------


def checkpoint_dir(
    root: str | os.PathLike, run: Run | str | None = None, shared: bool | None = None
) -> Path:
    """`<root>/<run_id>/`, created. Local run without an id: `<root>/local-<time>`.

    `shared` (default: `shared_checkpoints` in mlflow.toml, else False):
    `root` is shared by users whose tasks have no common group, as on an HPC
    volume. The directories this call creates up to and including `root`
    are then `1777`, so anyone adds a run and no one removes another's;
    otherwise they get the umask. The run's own directory belongs to whoever
    started the run. A run directory this process cannot write to is an
    error here, before training starts: start a new run from it instead
    (init_from)."""
    run_id = run.run_id if isinstance(run, Run) else run
    run_id = run_id or os.environ.get(ENV_RUN_ID) or time.strftime("local-%Y%m%d-%H%M%S")
    root = Path(root)
    if shared is None:
        shared = bool(project_config().get("shared_checkpoints", False))
    if shared:
        _mkdir_shared(root)
    else:
        root.mkdir(parents=True, exist_ok=True)
    path = root / run_id
    path.mkdir(exist_ok=True)
    if not os.access(path, os.W_OK | os.X_OK):
        raise PermissionError(
            f"{path} is not writable by this user (owner uid {path.stat().st_uid}): the run "
            f"was started by someone else. Start a new run from it: {ENV_INIT_FROM}={run_id}"
        )
    return path


def init_checkpoint_dir(root: str | os.PathLike, run: Run | None = None) -> Path | None:
    """`<root>/<init_from>/` of a new run that starts from another run's
    checkpoints, read-only to this run; None if the run has no init_from."""
    init_from = run.init_from if isinstance(run, Run) else os.environ.get(ENV_RUN_INIT)
    if not init_from:
        return None
    path = Path(root) / init_from
    return path if path.is_dir() else None


def prune_checkpoints(
    directory: str | os.PathLike,
    pattern: str = "*",
    keep_last: int = 2,
    keep: Iterable[str | os.PathLike] = (),
) -> list[Path]:
    """Delete all but the newest `keep_last` files matching `pattern` and those in `keep`.
    Returns what was deleted."""
    files = sorted(
        (p for p in Path(directory).glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    protected = {Path(p).resolve() for p in keep}
    removed = []
    for p in files[keep_last:]:
        if p.resolve() not in protected:
            try:
                p.unlink()
            except PermissionError as e:
                log.warning("checkpoint not removed: %s", e)
                continue
            removed.append(p)
    return removed


def log_checkpoint(path: str | os.PathLike, artifact_path: str = "checkpoints") -> str:
    """Upload a selected checkpoint as an artifact of the active run and tag its sha256.
    Returns the digest. Intermediate checkpoints stay on disk, not here."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    key = re.sub(r"[^\w./ -]", "_", f"{artifact_path}/{path.name}")
    mlflow.set_tag(f"sha256.{key}", digest)
    mlflow.log_artifact(str(path), artifact_path)
    return digest


# --- internals -----------------------------------------------------------------


def _repro_tags(sha: str | None, task_id: str | None, exp: str | None) -> dict[str, str]:
    tags: dict[str, str] = {}
    dirty = None
    if sha is None and os.environ.get(ENV_SHA):
        sha = os.environ[ENV_SHA]
        # A transport ships a committed tree, clean unless it says otherwise.
        dirty = os.environ.get(ENV_DIRTY, "false")
    paths: list[str] = []
    if sha is None:
        sha, paths = _git()
        dirty = None if sha is None else bool(paths)
    if sha:
        tags[TAG_SHA] = sha
        if dirty is not None:
            tags[TAG_DIRTY] = str(dirty).lower()
        if paths:
            tags[TAG_DIRTY_PATHS] = ", ".join(paths)[:5000]
    else:
        log.warning("no git commit for the run: pass sha= or set %s", ENV_SHA)
    task_id = task_id or os.environ.get(ENV_TASK_ID)
    if task_id:
        tags[TAG_TASK_ID] = task_id
    exp = exp or os.environ.get(ENV_EXP)
    if exp:
        tags[TAG_EXP] = exp
    return tags


def _user() -> str:
    """The server login, the same on every machine; else the OS user."""
    try:
        from mlflow.utils.credentials import read_mlflow_creds

        if name := read_mlflow_creds().username:
            return name
    except Exception:  # noqa: BLE001, S110 - an author is optional
        pass
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git() -> tuple[str | None, list[str]]:
    """HEAD and the uncommitted paths that could have produced the numbers.

    Untracked files count (a script not yet committed is code the commit does
    not have); .gitignore'd files and markdown do not, as in research-log's
    record_run.py, which exempts the log's documents.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, []
    paths = []
    for line in porcelain.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:  # a rename: the destination is what the tree has
            path = path.split(" -> ", 1)[1]
        if path and not path.endswith(".md"):
            paths.append(path)
    return sha, paths


def _mkdir_shared(root: Path) -> None:
    """Create `root` and its missing parents as `1777` (sticky, everyone writes)."""
    missing = []
    p = root
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        try:
            d.mkdir()
        except FileExistsError:  # created concurrently by another task
            continue
        try:
            d.chmod(0o1777)
        except OSError as e:
            log.warning("%s not made shared (1777): %s", d, e)


def _same_commit(client: MlflowClient, run_id: str, sha: str | None) -> bool:
    """A run is resumed only from its own commit; unknown on either side passes."""
    old = client.get_run(run_id).data.tags.get(TAG_SHA)
    if not sha or not old or old.startswith(sha) or sha.startswith(old):
        return True
    log.warning(
        "MLflow run %s was started from %s, this is %s: a new run starts from its checkpoints (%s)",
        run_id,
        old[:12],
        sha[:12],
        TAG_INIT_FROM,
    )
    return False


def _child_of_init_parent(
    client: MlflowClient, exp_id: str, tags: Mapping[str, str], match: Mapping[str, Any] | None
) -> str | None:
    """A child of a new sweep whose parent starts from an old sweep: the
    unfinished child of the old sweep with the same job number."""
    parent = tags.get(TAG_PARENT)
    if not parent:
        return None
    old_parent = client.get_run(parent).data.tags.get(TAG_INIT_FROM)
    if not old_parent:
        return None
    return _find_unfinished(client, exp_id, {**tags, TAG_PARENT: old_parent}, "child", match)


def _experiment_id(client: MlflowClient, name: str | None) -> str:
    if name is None:
        return mlflow.tracking.fluent._get_experiment_id() or "0"  # MLflow's Default
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    uri = mlflow.get_tracking_uri()
    location = None
    if uri.startswith(SQLITE):  # artifacts beside the database, not in whatever the cwd is
        location = (Path(uri.removeprefix(SQLITE)).resolve().parent / "mlruns").as_uri()
    try:
        return client.create_experiment(name, artifact_location=location)
    except Exception:  # created concurrently by another task
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise
        return exp.experiment_id


def _q(v: str) -> str:
    return "'" + str(v).replace("'", "\\'") + "'"


def _find_unfinished(
    client: MlflowClient,
    exp_id: str,
    tags: Mapping[str, str],
    role: str,
    match: Mapping[str, Any] | None,
) -> str | None:
    task_id = tags.get(TAG_TASK_ID)
    keys = {TAG_ROLE: role, **{k: str(v) for k, v in (match or {}).items()}}
    if role == "child":
        if TAG_PARENT not in tags:
            return None
        keys[TAG_PARENT] = tags[TAG_PARENT]
    elif task_id:
        keys[TAG_TASK_ID] = task_id
    else:
        return None  # a local top-level run resumes only by explicit id
    flt = " and ".join(f"tags.`{k}` = {_q(v)}" for k, v in keys.items())
    flt += " and attributes.status != 'FINISHED'"
    runs = client.search_runs([exp_id], flt, max_results=1, order_by=["attributes.start_time DESC"])
    return runs[0].info.run_id if runs else None


def _warn_once(what: str, e: Exception, _seen: dict[str, int] = {}) -> None:  # noqa: B006
    n = _seen.get(what, 0)
    _seen[what] = n + 1
    if n == 0 or n in (10, 100, 1000) or n % 10000 == 0:
        log.warning("MLflow %s failed (%d times so far), the run goes on: %s", what, n + 1, e)


_ORIGINAL: dict[str, Any] = {}


def _original(name: str) -> Any:
    if name not in _ORIGINAL:
        _ORIGINAL[name] = getattr(mlflow, name)
    return _ORIGINAL[name]


def _guard() -> None:
    for name in _GUARDED:
        if not hasattr(mlflow, name):
            continue
        fn = _original(name)

        @functools.wraps(fn)
        def wrapper(*a, _fn=fn, _name=name, **kw):
            try:
                return _fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                _warn_once(_name, e)
                return None

        setattr(mlflow, name, wrapper)


def _silence(why: str) -> None:
    log.info("MLflow logging disabled in this process (%s)", why)
    for name in _SILENCED:
        if hasattr(mlflow, name):
            _original(name)
            setattr(mlflow, name, lambda *a, **kw: None)


def _restore() -> None:
    """Undo _silence: the next run in this process (a sweep job) may succeed."""
    for name, fn in _ORIGINAL.items():
        setattr(mlflow, name, fn)


def _check_secrets() -> None:
    if os.environ.get(ENV_TASK_ID) and (
        os.environ.get("MLFLOW_TRACKING_PASSWORD") or os.environ.get("MLFLOW_TRACKING_TOKEN")
    ):
        log.warning(
            "MLflow credentials are in the task environment; they belong in "
            "~/.mlflow/credentials (the environment may be visible in the task description)"
        )
    cred = Path.home() / ".mlflow" / "credentials"
    try:
        mode = cred.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by others; run: chmod 600 %s", cred, cred)
