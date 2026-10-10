"""Lightning logger that writes into the run opened by tracking.start_run or
the Hydra callback, instead of creating its own.

Copied into the project from the mlflow-tracking skill, next to tracking.py.

    trainer = L.Trainer(logger=run_logger(), callbacks=checkpoints(ckpt_root))
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path(ckpt_root))

Differences from MLFlowLogger: the run's lifecycle belongs to the core, so
finalize() does not terminate it; logging is asynchronous and never raises;
hyperparameters already logged (by the Hydra callback) are not rewritten;
checkpoints are not uploaded (log_model=False), see tracking.log_checkpoint.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.utilities import rank_zero_only

from __pkg__ import tracking

log = logging.getLogger(__name__)


class RunLogger(MLFlowLogger):
    def __init__(self, run_id: str, experiment_id: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("tracking_uri", mlflow.get_tracking_uri())
        kwargs.setdefault("synchronous", False)
        kwargs["log_model"] = False
        super().__init__(run_id=run_id, **kwargs)
        self._experiment_id = experiment_id

    @property
    @rank_zero_experiment
    def experiment(self):
        # MLFlowLogger reads the run from the server here, at setup: with the
        # server down, training would not start. The run is known already.
        if not self._initialized:
            mlflow.set_tracking_uri(self._tracking_uri)
            self._initialized = True
        return self._mlflow_client

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        params = params if isinstance(params, dict) else vars(params)
        tracking.log_params(params, run_id=self.run_id)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        try:
            super().log_metrics(metrics, step)
        except Exception as e:  # noqa: BLE001
            tracking._warn_once("lightning log_metrics", e)

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        pass  # tracking.end_run / the Hydra callback ends the run


def run_logger(**kwargs: Any) -> RunLogger | DummyLogger:
    """A RunLogger bound to the active run. On rank != 0 the run id comes from
    rank 0 through the environment; without a run (not created) a DummyLogger,
    so that Lightning does not create a run of its own."""
    active = mlflow.active_run()
    run_id = active.info.run_id if active else os.environ.get(tracking.ENV_RUN_ID)
    if not run_id:
        log.warning("no MLflow run: Lightning logs nowhere")
        return DummyLogger()
    return RunLogger(run_id, active.info.experiment_id if active else None, **kwargs)


def checkpoints(
    ckpt_root: str | os.PathLike,
    monitor: str = "val_loss",
    mode: str = "min",
    every_n_epochs: int = 1,
) -> list[ModelCheckpoint]:
    """Checkpoints in <ckpt_root>/<run_id>/: the best by `monitor`, and the last
    two (last.ckpt among them) for resuming."""
    d = str(tracking.checkpoint_dir(ckpt_root))
    best = ModelCheckpoint(
        dirpath=d,
        filename="best-{epoch}-{step}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        every_n_epochs=every_n_epochs,
    )
    last = ModelCheckpoint(
        dirpath=d,
        filename="step-{step}",
        monitor="step",
        mode="max",
        save_top_k=2,
        save_last=True,
        every_n_epochs=every_n_epochs,
    )
    return [best, last]


def resume_path(ckpt_root: str | os.PathLike) -> str | None:
    """last.ckpt of the current run, if the run is a resumed one; else
    last.ckpt of the run it starts from (init_from); else None. Either way
    Lightning restores the full state: weights, optimizer, step."""
    p = tracking.checkpoint_dir(ckpt_root) / "last.ckpt"
    if p.is_file():
        return str(p)
    init = tracking.init_checkpoint_dir(ckpt_root)
    if init is not None and (init / "last.ckpt").is_file():
        return str(init / "last.ckpt")
    return None


def best_checkpoint(callbacks: list[ModelCheckpoint]) -> Path | None:
    path = callbacks[0].best_model_path
    return Path(path) if path else None
