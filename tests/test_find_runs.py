import subprocess
import sys
import uuid
from pathlib import Path

import find_runs
from mlflow.tracking import MlflowClient

from __pkg__ import tracking

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "find_runs.py"


def test_find_by_task_sha_exp(mlflow_env):
    task = f"task-{uuid.uuid4().hex[:6]}"
    sha = uuid.uuid4().hex
    name = f"f-{uuid.uuid4().hex[:6]}"
    parent = tracking.start_run(
        experiment=name, sha=sha, task_id=task, exp="E-0009.abc", role="parent", activate=False
    )
    for i in range(3):
        with tracking.start_run(
            experiment=name,
            sha=sha,
            task_id=task,
            role="child",
            parent_run_id=parent.run_id,
            tags={"hydra.job.num": i},
        ):
            pass
    tracking.terminate(parent.run_id)
    with tracking.start_run(experiment=name, sha=sha, task_id=task, exp="E-0009.abc"):
        pass

    client = MlflowClient()
    by_task = find_runs.find(client, task_id=task)
    assert [r["role"] for r in by_task] == ["parent", "single"]
    assert by_task[0]["run_id"] == parent.run_id
    assert by_task[0]["children"] == {"FINISHED": 3}
    assert [r["run_id"] for r in find_runs.find(client, sha=sha[:8])] == [
        r["run_id"] for r in by_task
    ]
    assert len(find_runs.find(client, exp="E-0009.abc", experiment=name)) == 2
    assert find_runs.find(client, task_id="nope") == []

    args = find_runs.record_run_args(by_task)
    assert args.startswith(f"--tracker mlflow --run-id {parent.run_id} --run-id ")
    assert f"--commit {sha}" in args and "--date " in args


def test_cli(mlflow_env):
    task = f"task-{uuid.uuid4().hex[:6]}"
    with tracking.start_run(experiment="cli", sha="deadbeef", task_id=task) as run:
        pass
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--task-id", task, "--tracking-uri", mlflow_env],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith(
        f"--tracker mlflow --run-id {run.run_id} --commit deadbeef --date "
    )
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--task-id", "none", "--tracking-uri", mlflow_env],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1 and "no runs found" in missing.stderr
