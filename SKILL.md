---
name: mlflow-tracking
description: Record a project's runs (training, molecular dynamics, benchmarks) in an MLflow tracking server, or a local SQLite store without one, so each run is found by commit, research-log experiment and transport task id; one parent run per Hydra sweep; a network failure costs metrics, not the computation; credentials in ~/.mlflow/credentials, never in the task environment; checkpoints on disk and resuming into the same run; only rank 0 writes under DDP. Use when a project logs to MLflow or should, when connecting a project to an MLflow server, when adding MLflow to a Hydra or PyTorch Lightning project, when a run must be resumed, to find the MLflow runs of a task, commit or experiment for research-log, or to read a run's metrics, compare a sweep's children or fetch its checkpoints, to give a person access to a project's MLflow experiment, or to change one's own MLflow password.
---

# mlflow-tracking

The code of a run logs with the plain MLflow API (`mlflow.log_metric`); the
run is already open, opened by the core (`tracking.start_run`) or by an
integration (Hydra callback). The skill knows nothing about a particular
project or where it runs: the same code works locally and in a task on a
remote machine.

| file | what | goes into the project |
|---|---|---|
| [`templates/__pkg__/tracking.py`](templates/__pkg__/tracking.py) | core: `start_run`, tags, rank guard, network resilience, checkpoints | always |
| [`templates/__pkg__/mlflow_hydra.py`](templates/__pkg__/mlflow_hydra.py) | Hydra callback: a run per job, a parent per sweep | with Hydra, [guide](integrations/hydra.md) |
| [`templates/__pkg__/mlflow_lightning.py`](templates/__pkg__/mlflow_lightning.py) | Lightning logger into the active run, checkpoints, resume | with Lightning, [guide](integrations/lightning.md) |
| [`scripts/find_runs.py`](scripts/find_runs.py) | find runs, print `record_run.py` arguments | stays in the skill |
| [`scripts/show_run.py`](scripts/show_run.py) | read a run: sweep table, metric history, checkpoints with sha256 check | stays in the skill |
| [`scripts/doctor.py`](scripts/doctor.py) | check config, credentials, login and network, never printing a secret | stays in the skill |
| [`templates/mlflow.toml`](templates/mlflow.toml) | server URI, experiment, S3 endpoint | always, at the project root |

Templates are copied, not installed as a package: a task sees only the
committed tree and the locked environment, and the project decides when to
update its copy. A template's docstring says where it came from. The
Python templates are a package named `__pkg__`, a valid identifier, so they
import each other as the copies will (`from __pkg__ import tracking`).

## Connecting a project

1. Copy `templates/__pkg__/tracking.py` (and the integrations the project
   uses) into the project's package, e.g. `src/<pkg>/tracking.py`, and
   replace `__pkg__` in the copies with the package name:
   `sed -i.bak 's/__pkg__/<pkg>/g' src/<pkg>/{tracking,mlflow_*}.py && rm src/<pkg>/*.bak`.
2. Add `mlflow-skinny` to the project's dependencies (the client without
   server packages); for a [local store](#local-store) also `sqlalchemy`
   and `alembic`; `hydra-core` and `lightning` only if used.
3. Copy `templates/mlflow.toml` to the project root and fill in the server
   URI, or `sqlite:///logs/mlflow.db` for a [local store](#local-store), and the
   MLflow experiment (`artifacts_endpoint` only with multipart upload,
   `shared_checkpoints` only on a shared volume, see
   [Checkpoints](#checkpoints)); commit it. `start_run`,
   the Hydra callback, `find_runs.py`, `show_run.py` and `doctor.py` read
   it from the current directory or above; an argument,
   `MLFLOW_TRACKING_URI` or `MLFLOW_EXPERIMENT_NAME` overrides it. Not in
   code and not in this skill. Nothing in it is a secret.
4. Credentials, for a server with a login: add the [access row](#secrets)
   to the project's list of access and tell the user which file to create.
   Then `doctor.py`.
5. Add to the project's `AGENTS.md` (or `CLAUDE.md`):
   `Runs are tracked with the mlflow-tracking skill (<path to tracking.py>, mlflow.toml).`

Without a framework (molecular dynamics, benchmarks):

```python
import mlflow
from mypkg import tracking

with tracking.start_run(experiment="md-benchmarks", params=cfg) as run:
    state_dir = tracking.checkpoint_dir(ckpt_root, run)  # restart files here
    for step in ...:
        mlflow.log_metric("energy", e, step=step)
```

With the Hydra callback, the task function does not call `start_run`: the
run is open when it starts.

## Local store

Without a server, `tracking_uri = "sqlite:///logs/mlflow.db"` in
`mlflow.toml` keeps everything in the project: the database at that path,
relative to `mlflow.toml` (not to the current directory; missing
directories are created), and artifacts in `mlruns/` beside it. Put it
where the project's other run outputs are (Hydra's `logs/` above), already
in `.gitignore`; cleaning that directory then deletes the run history too.
No login, no credentials file; `doctor.py` checks only that the database
is writable and that its packages are installed.

`mlflow-skinny` opens a sqlite store only with `sqlalchemy` and `alembic`
installed (`uv add mlflow-skinny sqlalchemy alembic`, or the full `mlflow`,
which has them). Without them `start_run` logs `MLflow run not created`
(unsupported URI `sqlite:///...`), the computation goes on unlogged, and
`find_runs.py` fails; `doctor.py` reports them missing.
Everything else in this skill works the same: tags, sweeps, resuming,
`find_runs.py`, `show_run.py`. `access.py` is for a server only.

The UI, from the project root, with the full `mlflow` package (the project
needs only `mlflow-skinny` and the two above):

```bash
uvx mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

A run on a remote machine writes to a database there, not to the laptop's:
a local store suits runs on one machine. Moving to a server later means a
new `tracking_uri`; the old runs stay in the file.

## Tags

Set by `start_run` on every run:

| tag | from |
|---|---|
| `git.sha`, `git.dirty` | argument, else `RUNNER_GIT_SHA` (+ `RUNNER_GIT_DIRTY`, default `false`), else `git` in the cwd; from `git`, dirty counts untracked files too (not `.gitignore`d ones, not `*.md`), as research-log's `record_run.py` does |
| `git.dirty.paths` | the uncommitted paths that made it dirty, from `git` |
| `runner.task_id` | argument or `RUNNER_TASK_ID`; absent for a local run |
| `exp` | argument or `RUNNER_EXP`: the research-log experiment id (`E-NNNN.xxx`), if any |
| `sweep.role` | `parent`, `child` or `single`; exists because MLflow cannot filter on a missing tag |
| `init_from.run_id` | a new run that starts from another run's checkpoints, see [Resuming](#resuming) |
| `mlflow.user` | the login in `~/.mlflow/credentials`, else the OS user; MLflow's run author, kept on resume |
| `mlflow.parentRunId` | children of a sweep |
| `hydra.job.num` | children of a sweep |

`RUNNER_*` are this skill's variables. A transport sets them from its own;
the core does not know any transport's variable names.

The data version needs no tag: a `.dvc` file in the commit pins it and
`git.sha` pins the commit.

## Finding runs

```bash
python3 <skill>/scripts/find_runs.py --task-id <id>     # a run through a transport
python3 <skill>/scripts/find_runs.py --sha <sha>        # a local run
python3 <skill>/scripts/find_runs.py --exp E-0007.k2f
```

It searches every experiment for runs with `sweep.role` `parent` or
`single`, prints a summary to stderr (status, children by status) and on
stdout the arguments for research-log's `record_run.py`:
`--tracker mlflow --run-id <id> [--run-id ...] --commit <sha> --date <day>`.
The journal gets the parent's id, or the single run's; children are found
by `mlflow.parentRunId`. It needs `mlflow-skinny`: run it with the
project's environment, or `uv run --with mlflow-skinny python3 ...` (for a
local store also `--with sqlalchemy --with alembic`). The
server: `--tracking-uri` or `MLFLOW_TRACKING_URI`.

Nothing is found when the run was never created (the server was down at
start). That is a visible error: the task log has `MLflow run not created`,
and there is nothing to record.

## Reading runs

A transport brings back its output directories (logs, Hydra's config), not
metrics: the numbers are in MLflow. Read them by run id, from
`find_runs.py`:

```bash
python3 <skill>/scripts/show_run.py <parent_id>                    # children of a sweep, one row each
python3 <skill>/scripts/show_run.py <parent_id> --metrics val_loss --sort val_loss
python3 <skill>/scripts/show_run.py <run_id>                       # tags, params, last metrics, artifacts
python3 <skill>/scripts/show_run.py <run_id> --history val_loss    # step,value,timestamp CSV
python3 <skill>/scripts/show_run.py <run_id> --download ckpt/      # checkpoints, checked by sha256
```

Same environment, credentials and server as `find_runs.py`; `--json` for
any of them.

- **Sweep table.** One row per child, ordered by `hydra.job.num`: status,
  the params that differ between children (the swept ones), and the
  metrics. A child that is not `FINISHED` has partial metrics; say so
  before comparing it.
- **Last, not best.** A run's metric value (in the table and in
  `get_run`) is the last logged one. The best epoch, a plateau or a
  divergence are in `--history` (`MlflowClient().get_metric_history`).
  Name which one a number is.
- **Checkpoints.** `--download` fetches `checkpoints/` (`--artifact-path`
  for another directory) and checks each file against its
  `sha256.checkpoints/<name>` tag: `ok`, `MISMATCH` (exit 2), or `no tag`
  (uploaded without `tracking.log_checkpoint`). Intermediate checkpoints
  are not in MLflow; they are on disk in `checkpoint_dir(root, run)`.
- **Anything else** (filters over metrics, other experiments, plots):
  `mlflow.search_runs(experiment_ids=..., filter_string=...)` returns a
  pandas DataFrame with `params.*`, `metrics.*`, `tags.*` columns.

What goes into research-log and what stays in MLflow:

- The record (an experiment or a technote) carries the numbers its
  conclusion rests on, copied in, each with its run id, metric name and
  which value (last, or best at step N). A run on the server can be
  deleted or renamed; the record cannot rely on it to make sense.
- The run ids go into `run.ids` through `record_run.py --run-id`: the
  parent of a sweep, not each child.
- Curves, full sweep tables, artifacts and checkpoints stay in MLflow; the
  record points to them by run id. A short table of the children it
  compares is fine; a dump of every metric is not.

## Secrets

A server with a login (basic auth) only; a local store has none.
`~/.mlflow/credentials`, mode 600, on every machine that logs (the laptop,
the home directory a remote task sees):

```ini
[mlflow]
mlflow_tracking_username = <login on the server>
mlflow_tracking_password = <its password>
```

MLflow reads it by itself. Use a personal login, so that the server knows
who made a run and an account can be revoked alone, with a password that
is the server's own, not a corporate one: administrators of a remote
machine can read the home directory there. A technical account only for
automation that no person runs (CI).

A user without admin rights changes their own password, for instance the
one an administrator sent in a message; the user runs this in their own
terminal, and the old password stops working at once (checked on a server
with basic auth, 2026-10-06):

```bash
printf 'Login: '; read -r U; printf 'Old password: '; read -rs OLD; echo; printf 'New password: '; read -rs NEW; echo; curl -fsS -u "$U:$OLD" -X PATCH "<server>/api/2.0/mlflow/users/update-password" -H 'Content-Type: application/json' -d "{\"username\":\"$U\",\"password\":\"$NEW\"}" && echo ok; unset OLD NEW
```

`{}` and `ok` mean done, 401 a wrong login or old password. No `"` or `\`
in the new password: they break the JSON. Then the new password goes into
the file here and in every remote home that has it.

`MLFLOW_TRACKING_USERNAME`/`PASSWORD` are not put into a task's
environment: a transport may show the environment in the task description,
and the variables override the file. `start_run` warns when it sees them in
a transport's task, and when the file is readable by others. Not in the
repository, not in `mlflow.toml`.

**The agent does not handle the secret.** It does not open the file, print
it, or pass a password in an argument or a variable. It learns the state
from `doctor.py`, which prints names, modes and the server's answers, and
when something is missing it shows the user the section above with
placeholders and the path; the user fills it in an editor. Delivering the
file to a remote home is the transport's (see [Transport](#transport)).

```bash
uv run python <skill>/scripts/doctor.py     # in the project; exit 1 if a run would not log
```

It checks `mlflow.toml`, the variables above, the file (section, keys,
mode), `/health`, the login against the experiment, and, if set, a TCP
connection to `artifacts_endpoint`. Whether the server has multipart upload turned on is
not visible to a client.

The project's row in its list of access (README), in the common format:

| access | file, section | keys | where | check |
|---|---|---|---|---|
| MLflow | `~/.mlflow/credentials` `[mlflow]` | `mlflow_tracking_username`, `mlflow_tracking_password` | laptop (`find_runs.py`, `show_run.py`); every remote home whose tasks log | `doctor.py`, there |

## Access to the experiment

With basic auth, every login has the server's `default_permission` on
everything (usually `READ`) and its own permissions on single experiments.
The creator of an experiment gets `MANAGE` on it, and `MANAGE` lets a login
grant permissions on that experiment without admin rights. So the project's
author gives people access, when asked "give X access to the project":

```bash
uv run python <skill>/scripts/access.py grant <login>                      # EDIT: create runs, log
uv run python <skill>/scripts/access.py grant <login> --permission MANAGE  # also grant, delete
uv run python <skill>/scripts/access.py revoke <login>
uv run python <skill>/scripts/access.py show <login>
```

In the project: the experiment and the server come from `mlflow.toml`, the
author's login from `~/.mlflow/credentials`. The login must exist (the
administrator creates it), and the experiment too (the first run creates
it). `EDIT` is enough for work; a second person with `MANAGE` keeps access
grantable when the author is away, but `MANAGE` also deletes the experiment.
A refusal prints what to do: no `MANAGE`, no such login, no experiment yet.

The API changed after MLflow 3.10 (tested: 3.10 has only
`/api/2.0/mlflow/experiments/permissions/*`, 3.16 only
`/api/3.0/mlflow/users/permissions/*`); `access.py` tries the new one and
falls back to the old. On the new one, `show` of another login is refused to all but administrators, and a
changed permission is a revoke and a grant.

## Network

`start_run` turns on asynchronous logging and wraps MLflow's logging
functions: an unreachable server is a warning in the log, the computation
goes on, metrics of the outage are lost.

- Server down at start: no run, MLflow's logging functions become no-ops
  in the process, so the code's `mlflow.log_metric` does not start a stray
  run. The next `start_run` in the process (the next job of a sweep) tries
  again.
- Server lost mid-run: metrics are dropped with a warning; `end_run` and
  artifact uploads warn instead of raising.
- Retries are MLflow's (`MLFLOW_HTTP_REQUEST_MAX_RETRIES`,
  `MLFLOW_HTTP_REQUEST_TIMEOUT`); with the defaults a dead server can delay
  `start_run` by minutes.

## Ranks

Under DDP the script runs in every rank. Only rank 0 opens a run: the rank
is the first of `RANK`, `LOCAL_RANK`, `SLURM_PROCID`, `JSM_NAMESPACE_RANK`
that is set, as in Lightning's `rank_zero_only`, and `NODE_RANK` must be 0.
Other ranks get an inactive handle whose `run_id` comes from rank 0
(`TRACKING_RUN_ID`, set before Lightning spawns them), and MLflow's logging
functions become no-ops there. A name imported before (`from mlflow import
log_metric`) is not covered: use `mlflow.log_metric`. Ranks started all at
once by `torchrun` do not learn the run id; broadcast it if they need it.

## Checkpoints

- Intermediate: on disk, `tracking.checkpoint_dir(root, run)` =
  `<root>/<run_id>/`; keep the last two and the best
  (`tracking.prune_checkpoints`, or two `ModelCheckpoint`s with
  Lightning). Not uploaded to MLflow. `root` is the project's, on a volume
  that survives the task.
- Shared `root`, `shared_checkpoints = true` in `mlflow.toml` (or
  `checkpoint_dir(..., shared=True)`): for a volume several users' tasks
  write to, when a task has only the user's own group, as on many HPC
  clusters, and a group-writable root admits its creator only.
  `checkpoint_dir` then creates `root` (and missing parents) as `1777`:
  every user adds runs, the sticky bit stops anyone from removing
  another's. Without it they get the umask. The run's directory belongs to
  whoever started the run; one this process cannot write to raises
  `PermissionError` in `checkpoint_dir`, before training, with the remedy
  (`RUNNER_INIT_FROM_RUN_ID`). `prune_checkpoints` skips files it cannot
  remove, with a warning.
- Selected: `tracking.log_checkpoint(path)` uploads it as an artifact and
  tags its sha256 (`sha256.checkpoints/<name>`).

On a server that serves artifacts (`mlflow-artifacts:/`) behind a proxy
with a request size limit (nginx), large artifacts pass only with
`MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true` on the server: parts of 100 MB
go to its S3 by presigned URL, bypassing the proxy. The server's
administrators set this up; the client needs no S3 key, but the S3
endpoint must be reachable from the task (`artifacts_endpoint` lets
`doctor.py` check it). The alternative, a server with
`--no-serve-artifacts`, makes the client write to S3 itself and every task
need an S3 key. A local store writes artifacts to disk.

## Resuming

`start_run` reopens a run instead of creating one:

- `resume_run_id=` or `RUNNER_RESUME_RUN_ID` (a top-level run; a sweep's
  children follow their parent);
- otherwise, for a run with `runner.task_id`: the latest unfinished run
  (`RUNNING`, `FAILED`, `KILLED`) with the same task id and role, and for
  a child the same parent and `hydra.job.num`.

A local run (no task id) is reopened only by explicit id. Params are
immutable in MLflow: a changed value keeps the old one, with a warning.
The code resumes its own state from `checkpoint_dir(root, run)`, which is
the same directory since the run id is the same.

**Only from the same commit.** A research-log record names one commit, so
one MLflow run must not mix two versions of the code. A run found for
resuming whose `git.sha` is not this commit (a prefix matches) is left as
it was, and a new run starts from it: tag `init_from.run_id`,
`run.init_from`, and its checkpoints in
`tracking.init_checkpoint_dir(root, run)` (read-only to the new run). A
sweep's children follow: a child of the new sweep starts from the
unfinished child of the old one with the same `hydra.job.num`. Unknown sha
on either side resumes as before.

**Starting from someone else's run**, or from a finished one: `init_from=`
or `RUNNER_INIT_FROM_RUN_ID`. Another user's run directory is not writable
to this user; that run is not resumed, a new one starts from it. With
Lightning, `resume_path(root)` returns the run's own `last.ckpt`, else the
`init_from` run's, and `fit(ckpt_path=...)` restores the full state; a
project that wants the weights only loads them itself from
`init_checkpoint_dir`.

Whether a transport's restart keeps the task id decides whether the
automatic match works; if the id changes, the transport (or the user)
passes `RUNNER_RESUME_RUN_ID`.

## Transport

A transport delivers a task to a remote machine and brings results back.
Locally none of this is needed. A transport that supports this skill:

1. Puts `~/.mlflow/credentials` into the home directory on that machine,
   once: without passing the password in the task environment, without
   overwriting what the file already has, and without leaving a copy in
   its own storage (task inputs are often kept). Then runs `doctor.py`
   in a task, which checks that home and the network from there.
2. Sets `RUNNER_GIT_SHA` (the committed tree it ships),
   `RUNNER_TASK_ID` (the id the user later fetches results by) and, if
   given, `RUNNER_EXP`.
3. Points Hydra's output directories (`hydra.run.dir`, `hydra.sweep.dir`)
   to where it collects results from.
4. After fetching results, offers `find_runs.py --task-id <id>` for
   recording the run: the transport's task id goes into the experiment's
   text, the MLflow run id into `--run-id`.

Transports describe this in a recipe of their own; this skill does not
name them.

## research-log

`record_run.py` already takes `--tracker mlflow` and a repeatable
`--run-id`; `find_runs.py` prints both. The research-log experiment id
reaches the run as `RUNNER_EXP` (or `exp=`), and `find_runs.py --exp`
finds its runs back.
