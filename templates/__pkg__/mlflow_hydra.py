"""Hydra callback: one MLflow run per job, one parent run per sweep.

Copied into the project from the mlflow-tracking skill, next to tracking.py.
Enabled in the project's primary config:

    hydra:
      callbacks:
        mlflow:
          _target_: <package>.mlflow_hydra.MlflowCallback
          # tracking_uri, experiment: default from mlflow.toml

The task function then logs with the plain MLflow API into the active run.
"""

from __future__ import annotations

import atexit
import copy
import functools
import logging
import os
from pathlib import Path
from typing import Any, cast

from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

from __pkg__ import tracking

log = logging.getLogger(__name__)

# The parent's id for jobs that run in other processes (joblib, submitit).
ENV_PARENT = "TRACKING_PARENT_RUN_ID"


def _never_raise(method):
    """Hydra turns a callback's exception into a warning and runs the job anyway;
    the job's mlflow.log_metric would then start a stray run. Mute MLflow instead."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow callback %s failed, logging disabled: %r", method.__name__, e)
            tracking._silence(f"callback {method.__name__} failed")

    return wrapper


def task_config(config: DictConfig) -> dict:
    """The job's config without the hydra node, resolved as the task function sees it."""
    cfg = copy.deepcopy(config)
    with read_write(cfg), open_dict(cfg):
        cfg.pop("hydra", None)
    try:
        return cast(dict, OmegaConf.to_container(cfg, resolve=True))
    except Exception:  # noqa: BLE001 - a missing value must not cost the run
        return cast(dict, OmegaConf.to_container(cfg, resolve=False))


class MlflowCallback(Callback):
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment: str | None = None,
        log_config: bool = True,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.log_config = log_config
        self.parent_id: str | None = None

    # --- sweep -------------------------------------------------------------

    @_never_raise
    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        hc = config.hydra
        params = {"overrides": " ".join(hc.overrides.task)}
        sweeper = cast(dict, OmegaConf.to_container(hc.sweeper, resolve=True))
        params.update({f"sweeper.{k}": v for k, v in sweeper.items() if k != "_target_"})
        parent = tracking.start_run(
            experiment=self.experiment,
            tracking_uri=self.tracking_uri,
            role="parent",
            run_name=hc.job.name,
            params=params,
            activate=False,
        )
        self.parent_id = parent.run_id if parent.active else None
        if self.parent_id:
            os.environ[ENV_PARENT] = self.parent_id
            # When a job fails, the sweeper raises after the batch and
            # on_multirun_end is never called: close the parent at exit.
            atexit.register(self._close_parent)

    @_never_raise
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        self._close_parent()

    # --- job ---------------------------------------------------------------

    @_never_raise
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        hc = config.hydra
        parent = self.parent_id or os.environ.get(ENV_PARENT)
        tags: dict[str, Any] = {}
        if hc.mode == RunMode.MULTIRUN and parent:
            role, num = "child", hc.job.num
            tags["hydra.job.num"] = num
            name = hc.job.override_dirname or f"{hc.job.name}-{num}"
            match = {"hydra.job.num": num}
        else:
            role, name, match, parent = "single", hc.job.name, None, None
        params = None
        if self.log_config:
            params = task_config(config)
        run = tracking.start_run(
            experiment=self.experiment,
            tracking_uri=self.tracking_uri,
            role=role,
            parent_run_id=parent,
            run_name=name,
            tags=tags,
            params=params,
            resume_match=match,
        )
        if run.active and self.log_config and hc.output_subdir:
            import mlflow

            subdir = Path(hc.runtime.output_dir) / hc.output_subdir
            for f in ("config.yaml", "overrides.yaml"):
                if (subdir / f).is_file():
                    mlflow.log_artifact(str(subdir / f), "hydra")

    @_never_raise
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        ok = job_return.status == JobStatus.COMPLETED
        tracking.end_run("FINISHED" if ok else "FAILED")

    # --- internals -----------------------------------------------------------

    def _close_parent(self) -> None:
        os.environ.pop(ENV_PARENT, None)
        if self.parent_id:
            parent, self.parent_id = self.parent_id, None
            tracking.terminate(parent, self._sweep_status(parent))

    def _sweep_status(self, parent_id: str) -> str:
        from mlflow.tracking import MlflowClient

        try:
            runs = MlflowClient().search_runs(
                [MlflowClient().get_run(parent_id).info.experiment_id],
                f"tags.`mlflow.parentRunId` = '{parent_id}'",
                max_results=10000,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow: sweep children not read, parent marked FINISHED: %s", e)
            return "FINISHED"
        return "FAILED" if any(r.info.status != "FINISHED" for r in runs) else "FINISHED"
