# Hydra

[`templates/__pkg__/mlflow_hydra.py`](../templates/__pkg__/mlflow_hydra.py): a
`hydra.experimental.callback.Callback` that opens the run before the task
function starts and closes it after. Copy it next to `tracking.py` and enable
it in the primary config:

```yaml
hydra:
  callbacks:
    mlflow:
      _target_: mypkg.mlflow_hydra.MlflowCallback
      # tracking_uri, experiment: default from mlflow.toml
      log_config: true        # the resolved config as params, config.yaml and overrides.yaml as artifacts
```

The task function logs with `mlflow.log_metric` and does not call
`start_run`.

| event | what it does |
|---|---|
| `on_multirun_start` | parent run (`sweep.role=parent`), not active in the main process; params: the overrides and the sweeper's settings |
| `on_job_start` | `start_run` of the job: in a sweep `mlflow.parentRunId`, `sweep.role=child`, `hydra.job.num`, run name = `hydra.job.override_dirname`; the resolved config without `hydra` as params, `.hydra/config.yaml` and `overrides.yaml` as artifacts under `hydra/` |
| `on_job_end` | `FINISHED` or `FAILED` from `JobReturn` |
| `on_multirun_end` | closes the parent: `FAILED` if any child did not finish |

Without `--multirun`: one run, `sweep.role=single`.

## How it behaves (tested, Hydra 1.3)

- **The parent link is an explicit tag**, not `start_run(nested=True)`:
  nesting works within one process, and the joblib and submitit launchers
  run jobs in others. The parent id reaches them through
  `TRACKING_PARENT_RUN_ID`, set before the launcher starts its workers.
- **joblib launcher:** `on_job_start` and `on_job_end` run in the worker
  process, together with the job, so the job's run is active where the task
  function runs.
- **A failing job does not stop the batch:** the basic launcher runs every
  job, then the sweeper raises and `on_multirun_end` is never called. The
  callback closes the parent at process exit instead (`atexit`), `FAILED`.
  A process killed by SIGKILL leaves the parent `RUNNING`.
- **Hydra swallows a callback's exception** (a `UserWarning`) and runs the
  job anyway; the job's `mlflow.log_metric` would then start a stray run in
  the default experiment. Every callback method therefore catches its own
  errors and mutes MLflow in the process.
- **Resuming a sweep:** with the same `RUNNER_TASK_ID`, the unfinished
  parent is reopened, and each child is matched by parent and
  `hydra.job.num`: failed jobs continue in their runs, finished ones get
  new runs.
- **Resuming from another commit:** the old sweep is left as it was; a new
  parent starts from it (`init_from.run_id`), and each new child from the
  old unfinished child with its `hydra.job.num`. Tested.

## DDP and --multirun

Lightning's subprocess launcher (`strategy="ddp"`) re-runs the script for
ranks 1..N-1 with `sys.argv[1:]` plus `hydra.run.dir=...`
(`lightning/fabric/strategies/launchers/subprocess_script.py`,
`_hydra_subprocess_cmd`). Without `-m` this is fine: the children are
single runs, see [lightning.md](lightning.md). With `-m` the arguments
still contain `-m`, and each child runs the whole sweep again. This is
Lightning's limitation, not the callback's. For DDP sweeps: one sweep point
per transport task, or `strategy="ddp_spawn"` (not tested here).
