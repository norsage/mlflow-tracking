# mlflow-tracking

A skill for Claude Code and other agents: record a project's runs in an MLflow
tracking server, or in a local SQLite store without one. Each run carries the commit, the research-log experiment and
the transport's task id; a Hydra sweep is one parent run with a child per job;
a network failure costs metrics, not the computation; credentials stay in
`~/.mlflow/credentials`; an interrupted run resumes into the same MLflow run;
under DDP only rank 0 writes. See [SKILL.md](SKILL.md); Hydra and Lightning
are in [integrations/](integrations/).

## Install

```sh
bash install.sh            # every project on this machine
bash install.sh --symlink  # link the checkout: the agent reads your working copy

cd my-project
bash /path/to/mlflow-tracking/install.sh --project  # into the current directory, committed with this project
```

Re-running replaces the install, after a prompt. A copy leaves out the test
setup.

## Connecting a project

1. Copy [`templates/__pkg__/tracking.py`](templates/__pkg__/tracking.py), and
   `mlflow_hydra.py` / `mlflow_lightning.py` if the project uses Hydra or
   Lightning, into the project's package, replacing `__pkg__` in them with
   the package name.
2. Add `mlflow-skinny` to the project's dependencies, and for a local store
   also `sqlalchemy` and `alembic`.
3. Copy [`templates/mlflow.toml`](templates/mlflow.toml) to the project
   root: server URI and experiment. Committed, no secrets. Without a server,
   `tracking_uri = "sqlite:///logs/mlflow.db"` (SKILL.md, "Local store").
4. For a server with a login, create `~/.mlflow/credentials` (mode 600)
   with your own login on it. Then check with
   `uv run python <skill>/scripts/doctor.py`.
5. Add to the project's `AGENTS.md` (or `CLAUDE.md`):
   `Runs are tracked with the mlflow-tracking skill (<path to tracking.py>, mlflow.toml).`

## Development

```sh
uv sync --group lightning   # torch and lightning only for tests/test_lightning.py
uv run pytest
```

The tests start a local MLflow server (sqlite, served artifacts) and run
Hydra and Lightning apps as subprocesses, DDP on CPU included. Without the
`lightning` group the Lightning tests are skipped.
