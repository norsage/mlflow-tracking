import json
import subprocess
import sys
import uuid
from pathlib import Path

import mlflow
import show_run
from mlflow.tracking import MlflowClient

from __pkg__ import tracking

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "show_run.py"


def _cli(uri, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--tracking-uri", uri],
        capture_output=True,
        text=True,
        check=False,
    )


def _sweep(name):
    parent = tracking.start_run(experiment=name, sha="abc", role="parent", activate=False)
    for i, lr in enumerate([0.1, 0.01, 0.001]):
        with tracking.start_run(
            experiment=name,
            sha="abc",
            role="child",
            parent_run_id=parent.run_id,
            params={"lr": lr, "epochs": 2},
            tags={"hydra.job.num": i},
        ):
            for step in range(3):
                mlflow.log_metric("loss", lr * 10 + 1 - step * 0.1, step=step)
            mlflow.log_metric("acc", 0.5 + i * 0.1)
    tracking.terminate(parent.run_id)
    mlflow.flush_async_logging()
    return parent


def test_children_table(mlflow_env):
    parent = _sweep(f"s-{uuid.uuid4().hex[:6]}")
    client = MlflowClient()

    rows, varying = show_run.children(client, parent.run_id)
    assert varying == ["lr"]  # epochs is the same everywhere
    assert [r["job"] for r in rows] == [0, 1, 2]
    assert rows[0]["params"] == {"lr": "0.1"}
    assert rows[0]["metrics"] == {"acc": 0.5, "loss": 1.8}  # last step of loss

    rows, _ = show_run.children(client, parent.run_id, ["loss"], sort="loss")
    assert [r["job"] for r in rows] == [2, 1, 0]
    assert list(rows[0]["metrics"]) == ["loss"]

    out = _cli(mlflow_env, parent.run_id, "--metrics", "loss", "--sort", "loss")
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0].split() == ["job", "status", "run_id", "lr", "loss"]
    assert lines[1].split()[0] == "2" and lines[1].split()[-1] == "0.81"
    assert parent.run_id in out.stderr

    data = json.loads(_cli(mlflow_env, parent.run_id, "--json").stdout)
    assert data["tags"]["sweep.role"] == "parent" and len(data["children"]) == 3


def test_history_and_summary(mlflow_env):
    parent = _sweep(f"h-{uuid.uuid4().hex[:6]}")
    child = show_run.children(MlflowClient(), parent.run_id)[0][0]["run_id"]

    out = _cli(mlflow_env, child, "--history", "loss")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[0] == "step,value,timestamp"
    assert [line.split(",")[:2] for line in out.stdout.splitlines()[1:]] == [
        ["0", "2.0"],
        ["1", "1.9"],
        ["2", "1.8"],
    ]
    assert _cli(mlflow_env, child, "--history", "nope").returncode == 1

    s = show_run.summary(MlflowClient(), child)
    assert s["tags"]["mlflow.parentRunId"] == parent.run_id
    assert s["tags"]["sweep.role"] == "child" and s["params"]["lr"] == "0.1"
    text = _cli(mlflow_env, child).stdout
    assert "loss = 1.8" in text and "git.sha = abc" in text


def test_download_checks_sha256(mlflow_env, tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"weights" * 1000)
    other = tmp_path / "notes.txt"
    other.write_text("x")
    with tracking.start_run(experiment=f"d-{uuid.uuid4().hex[:6]}", sha="abc") as run:
        tracking.log_checkpoint(ckpt)
        mlflow.log_artifact(str(other), "checkpoints")  # no tag
    client = MlflowClient()

    res = show_run.download_checkpoints(client, run.run_id, tmp_path / "dl")
    assert {Path(p).name: s for p, s in res.items()} == {"best.ckpt": "ok", "notes.txt": "no tag"}
    assert list(show_run.summary(client, run.run_id)["sha256"]) == ["checkpoints/best.ckpt"]

    client.set_tag(run.run_id, "sha256.checkpoints/best.ckpt", "0" * 64)
    out = _cli(mlflow_env, run.run_id, "--download", str(tmp_path / "dl2"))
    assert out.returncode == 2
    assert "MISMATCH" in out.stdout and "best.ckpt" in out.stdout
