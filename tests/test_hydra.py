"""Hydra apps in a temp dir, run as subprocesses like a user would."""

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"

APP = """
import os
import hydra
import mlflow

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    mlflow.log_metric("x2", cfg.x * 2)
    with open(os.path.join(os.environ["PIDS"], str(cfg.x)), "w") as f:
        f.write(str(os.getpid()))
    if cfg.x == cfg.fail_on:
        raise RuntimeError("job failed on purpose")

if __name__ == "__main__":
    main()
"""

CONFIG = """
x: 1
fail_on: -1
model:
  lr: 0.01
  layers: [1, 2]
hydra:
  callbacks:
    mlflow:
      _target_: __pkg__.mlflow_hydra.MlflowCallback
      experiment: {experiment}
"""


def make_app(tmp_path, experiment):
    shutil.copytree(TEMPLATES / "__pkg__", tmp_path / "__pkg__")
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "config.yaml").write_text(CONFIG.format(experiment=experiment))
    (tmp_path / "pids").mkdir()


def run_app(tmp_path, *args, env_extra=None):
    import os

    env = {
        **os.environ,
        "PIDS": str(tmp_path / "pids"),
        "RUNNER_GIT_SHA": "cafe",
        "RUNNER_TASK_ID": "hydra-task",
        **(env_extra or {}),
    }
    return subprocess.run(
        [sys.executable, "app.py", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def runs_of(experiment):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    return client.search_runs([exp.experiment_id], "", max_results=100)


def test_single_run(mlflow_env, tmp_path):
    name = f"h-{uuid.uuid4().hex[:6]}"
    make_app(tmp_path, name)
    out = run_app(tmp_path, "x=5")
    assert out.returncode == 0, out.stderr
    (run,) = runs_of(name)
    assert run.data.tags["sweep.role"] == "single"
    assert run.data.tags["git.sha"] == "cafe"
    assert run.info.status == "FINISHED"
    assert run.data.metrics["x2"] == 10
    assert run.data.params["x"] == "5" and run.data.params["model.lr"] == "0.01"
    assert "hydra.job.num" not in run.data.tags
    arts = {a.path for a in MlflowClient().list_artifacts(run.info.run_id, "hydra")}
    assert arts == {"hydra/config.yaml", "hydra/overrides.yaml"}


@pytest.mark.parametrize("launcher", ["basic", "joblib"])
def test_sweep_with_failed_job(mlflow_env, tmp_path, launcher):
    name = f"h-{uuid.uuid4().hex[:6]}"
    make_app(tmp_path, name)
    args = ["-m", "x=1,2,3", "fail_on=2"]
    if launcher == "joblib":
        args += ["hydra/launcher=joblib", "hydra.launcher.n_jobs=2"]
    out = run_app(tmp_path, *args)
    runs = runs_of(name)
    parents = [r for r in runs if r.data.tags["sweep.role"] == "parent"]
    children = [r for r in runs if r.data.tags["sweep.role"] == "child"]
    assert len(parents) == 1 and len(runs) == 4, [r.data.tags for r in runs]
    parent = parents[0]
    by_num = {int(c.data.tags["hydra.job.num"]): c for c in children}
    assert sorted(by_num) == [0, 1, 2]
    assert all(c.data.tags["mlflow.parentRunId"] == parent.info.run_id for c in children)
    assert all(c.data.tags["runner.task_id"] == "hydra-task" for c in children)
    assert by_num[1].info.status == "FAILED"
    assert by_num[0].info.status == by_num[2].info.status == "FINISHED"
    assert by_num[2].data.metrics["x2"] == 6
    assert by_num[2].info.run_name == "fail_on=2,x=3"
    assert parent.data.params["overrides"] == "x=1,2,3 fail_on=2"
    assert out.returncode == 1
    assert parent.info.status == "FAILED"
    pids = {p.read_text() for p in (tmp_path / "pids").iterdir()}
    # record for the task notes: exit code, parent status, process layout
    print(
        f"\n[{launcher}] exit={out.returncode} parent={parent.info.status} "
        f"distinct job pids={len(pids)}"
    )
    print(out.stderr[-600:])


def test_sweep_all_ok_parent_finished(mlflow_env, tmp_path):
    name = f"h-{uuid.uuid4().hex[:6]}"
    make_app(tmp_path, name)
    out = run_app(tmp_path, "-m", "x=1,2")
    assert out.returncode == 0, out.stderr
    runs = runs_of(name)
    assert len(runs) == 3 and all(r.info.status == "FINISHED" for r in runs)


def test_sweep_resumed_by_task_id(mlflow_env, tmp_path):
    """A transport restarts the same task: parent and children are reopened."""
    name = f"h-{uuid.uuid4().hex[:6]}"
    make_app(tmp_path, name)
    env = {"RUNNER_TASK_ID": f"t-{uuid.uuid4().hex[:6]}"}
    run_app(tmp_path, "-m", "x=1,2", "fail_on=2", env_extra=env)
    before = {r.info.run_id: r.info.status for r in runs_of(name)}
    out = run_app(tmp_path, "-m", "x=1,2", env_extra=env)
    assert out.returncode == 0, out.stderr
    after = runs_of(name)
    failed = [rid for rid, st in before.items() if st == "FAILED"]
    # the failed parent and the failed child are the same runs, now finished
    assert all(r.info.status == "FINISHED" for r in after if r.info.run_id in failed)
    parents = [r for r in after if r.data.tags["sweep.role"] == "parent"]
    assert len(parents) == 1
    # job 0 had finished: a new run for it; job 1 had failed: reopened
    assert len(after) == 4


def test_sweep_resumed_from_another_commit(mlflow_env, tmp_path):
    """Resumed from a new commit: a new sweep, each child starts from the
    old child with its job number that did not finish."""
    name = f"h-{uuid.uuid4().hex[:6]}"
    make_app(tmp_path, name)
    env = {"RUNNER_TASK_ID": f"t-{uuid.uuid4().hex[:6]}"}
    run_app(tmp_path, "-m", "x=1,2", "fail_on=2", env_extra=env)
    old = runs_of(name)
    old_parent = next(r for r in old if r.data.tags["sweep.role"] == "parent")
    old_failed = next(r for r in old if r.data.tags.get("hydra.job.num") == "1")
    out = run_app(
        tmp_path,
        "-m",
        "x=1,2",
        env_extra={**env, "RUNNER_GIT_SHA": "beef", "RUNNER_RESUME_RUN_ID": old_parent.info.run_id},
    )
    assert out.returncode == 0, out.stderr
    new = [r for r in runs_of(name) if r.data.tags.get("git.sha") == "beef"]
    assert len(new) == 3 and all(r.info.status == "FINISHED" for r in new)
    by_num = {r.data.tags.get("hydra.job.num"): r for r in new}
    assert by_num[None].data.tags["init_from.run_id"] == old_parent.info.run_id
    assert by_num["1"].data.tags["init_from.run_id"] == old_failed.info.run_id
    assert "init_from.run_id" not in by_num["0"].data.tags  # job 0 had finished


def test_callback_failure_mutes_mlflow(mlflow_env, monkeypatch):
    """Hydra only warns when a callback raises; the job must not start a stray run."""
    import mlflow

    sys.path.insert(0, str(TEMPLATES))
    from __pkg__ import mlflow_hydra

    def boom(*a, **kw):
        raise RuntimeError("callback bug")

    monkeypatch.setattr(mlflow_hydra.tracking, "start_run", boom)
    cb = mlflow_hydra.MlflowCallback(experiment="unused")
    cb.on_job_start(config=None)  # does not raise
    name = f"stray-{uuid.uuid4().hex[:6]}"
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", name)
    mlflow.log_metric("x", 1.0)
    assert mlflow.active_run() is None
    assert MlflowClient().get_experiment_by_name(name) is None
