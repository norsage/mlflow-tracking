# PyTorch Lightning

[`templates/__pkg__/mlflow_lightning.py`](../templates/__pkg__/mlflow_lightning.py), next
to `tracking.py`:

```python
import lightning as L
from mypkg import mlflow_lightning as ml, tracking

# the run is open: tracking.start_run(...) or the Hydra callback
callbacks = ml.checkpoints(cfg.ckpt_root, monitor="val_loss")
trainer = L.Trainer(logger=ml.run_logger(), callbacks=callbacks, ...)
trainer.fit(model, datamodule=dm, ckpt_path=ml.resume_path(cfg.ckpt_root))
tracking.log_checkpoint(ml.best_checkpoint(callbacks))   # the selected one goes to MLflow
```

## run_logger()

A `MLFlowLogger` subclass bound to the active run (`run_id=`), not a run
of its own. What it changes, from reading `lightning/pytorch/loggers/mlflow.py`
(Lightning 2.6) and tested:

| MLFlowLogger | RunLogger |
|---|---|
| `experiment` calls `get_run` at `setup`: with the server down, `fit` fails before the first step | knows the run, no request |
| `finalize()` terminates the run at the end of `fit` (and `test`), synchronously | does nothing: the core or the Hydra callback ends the run |
| `log_hyperparams` rewrites params: a value the Hydra callback logged differently is an error | `tracking.log_params`: existing params are kept |
| `log_model=True` uploads every selected checkpoint | `log_model=False`; upload the chosen one with `tracking.log_checkpoint` |
| synchronous `log_batch` | `synchronous=False`, errors are warnings |

Without a run (the server was down at start) `run_logger()` returns a
`DummyLogger`, so that Lightning does not create a run of its own.

## Ranks

Lightning's own methods are `rank_zero_only`. The script itself runs in
every rank: the subprocess launcher re-runs it for ranks 1..N-1 with
`LOCAL_RANK` already in the environment
(`lightning/fabric/strategies/launchers/subprocess_script.py`), so
`start_run` and the Hydra callback see the rank from the first line and do
not open a run there. Rank 0 exports `TRACKING_RUN_ID` before `fit`, so
the children find the same run and the same checkpoint directory. Tested:
DDP on CPU, 2 ranks, under the Hydra callback, gives one run.

`--multirun` with `strategy="ddp"` does not work, see [hydra.md](hydra.md#ddp-and---multirun).

## Checkpoints and resuming

`checkpoints(root)` returns two `ModelCheckpoint`s writing to
`<root>/<run_id>/`: the best by `monitor` (`save_top_k=1`) and the last two
by step, with `last.ckpt`. `resume_path(root)` is that `last.ckpt` if it
exists, which happens when `start_run` reopened the run (same
`runner.task_id`, or `RUNNER_RESUME_RUN_ID`). Metrics continue in the same
run, with Lightning's steps continuing from the checkpoint.

A run from another commit is not reopened: `start_run` creates a new run
with `init_from`, and `resume_path(root)` then returns the old run's
`last.ckpt` from `<root>/<init_from>/`. The new run writes its own
checkpoints in `<root>/<new run_id>/`; its metrics start at the restored
step.
