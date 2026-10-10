"""Lightning integration: a tiny model on CPU, a real MLflow server."""

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

L = pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from mlflow.tracking import MlflowClient

from __pkg__ import tracking

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
sys.path.insert(0, str(TEMPLATES))
from __pkg__ import mlflow_lightning

MODEL = """
import torch
import lightning as L


class Tiny(L.LightningModule):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Linear(4, 1)

    def training_step(self, batch, _):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.net(x), y), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def loader():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, x.sum(1, keepdim=True)),
                                       batch_size=16)
"""


@pytest.fixture
def model_module(tmp_path, monkeypatch):
    (tmp_path / "tiny.py").write_text(MODEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    import tiny

    return importlib.reload(tiny)


def fit(tiny, root, epochs, ckpt_path=None):
    cbs = mlflow_lightning.checkpoints(root)
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=mlflow_lightning.run_logger(),
        callbacks=cbs,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(tiny.Tiny(lr=0.05), tiny.loader(), tiny.loader(), ckpt_path=ckpt_path)
    return cbs


def test_logger_writes_into_core_run(mlflow_env, tmp_path, model_module):
    name = f"l-{uuid.uuid4().hex[:6]}"
    root = tmp_path / "ckpt"
    with tracking.start_run(experiment=name, sha="s", params={"lr": 0.05}) as run:
        cbs = fit(model_module, root, epochs=3)
        # finalize() did not end the run
        assert MlflowClient().get_run(run.run_id).info.status == "RUNNING"
        best = mlflow_lightning.best_checkpoint(cbs)
        tracking.log_checkpoint(best)
    r = MlflowClient().get_run(run.run_id)
    assert r.info.status == "FINISHED"
    assert "train_loss" in r.data.metrics and "val_loss" in r.data.metrics
    assert r.data.params["lr"] == "0.05"
    exp = MlflowClient().get_experiment_by_name(name)
    assert len(MlflowClient().search_runs([exp.experiment_id])) == 1
    files = sorted(p.name for p in (root / run.run_id).iterdir())
    assert "last.ckpt" in files and sum(f.startswith("best-") for f in files) == 1
    assert sum(f.startswith("step-") for f in files) == 2
    assert [a.path for a in MlflowClient().list_artifacts(run.run_id, "checkpoints")] == [
        f"checkpoints/{best.name}"
    ]


def test_resume_continues_run_and_training(mlflow_env, tmp_path, model_module, monkeypatch):
    name = f"l-{uuid.uuid4().hex[:6]}"
    root = tmp_path / "ckpt"
    monkeypatch.setenv("RUNNER_TASK_ID", f"t-{uuid.uuid4().hex[:6]}")
    run = tracking.start_run(experiment=name, sha="s")
    assert mlflow_lightning.resume_path(root) is None
    fit(model_module, root, epochs=2)
    tracking.end_run("FAILED")  # the task was interrupted
    with tracking.start_run(experiment=name, sha="s") as again:
        assert again.run_id == run.run_id
        path = mlflow_lightning.resume_path(root)
        assert path and Path(path).parent.name == run.run_id
        fit(model_module, root, epochs=4, ckpt_path=path)
    hist = MlflowClient().get_metric_history(run.run_id, "epoch")
    assert max(m.value for m in hist) == 3


def test_new_commit_starts_from_old_checkpoint(mlflow_env, tmp_path, model_module, monkeypatch):
    name = f"l-{uuid.uuid4().hex[:6]}"
    root = tmp_path / "ckpt"
    run = tracking.start_run(experiment=name, sha="s")
    fit(model_module, root, epochs=2)
    tracking.end_run("FAILED")
    with tracking.start_run(experiment=name, sha="t", resume_run_id=run.run_id) as new:
        assert new.init_from == run.run_id
        path = mlflow_lightning.resume_path(root)
        assert path and Path(path).parent.name == run.run_id
        fit(model_module, root, epochs=4, ckpt_path=path)
        assert (root / new.run_id / "last.ckpt").is_file()
    hist = MlflowClient().get_metric_history(new.run_id, "epoch")
    assert min(m.value for m in hist) == 2 and max(m.value for m in hist) == 3


def test_no_run_means_no_stray_run(mlflow_env, tmp_path, model_module):
    dead = "http://127.0.0.1:9"
    name = f"l-{uuid.uuid4().hex[:6]}"
    with tracking.start_run(experiment=name, sha="s", tracking_uri=dead) as run:
        assert not run.active
        assert isinstance(mlflow_lightning.run_logger(), L.pytorch.loggers.logger.DummyLogger)
        fit(model_module, tmp_path / "ckpt", epochs=1)


def test_server_lost_during_training(mlflow_env, tmp_path, tmp_path_factory, model_module):
    from conftest import start_server, stop_server

    proc, uri = start_server(tmp_path_factory.mktemp("lost-l"))
    with tracking.start_run(experiment="l-lost", sha="s", tracking_uri=uri):
        stop_server(proc)
        fit(model_module, tmp_path / "ckpt", epochs=2)


APP = """
import os
import hydra
import lightning as L
import mlflow
from __pkg__ import mlflow_lightning
from tiny import Tiny, loader

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    trainer = L.Trainer(max_epochs=2, logger=mlflow_lightning.run_logger(),
                        callbacks=mlflow_lightning.checkpoints(cfg.ckpt_root),
                        accelerator="cpu", devices=2, strategy="ddp", log_every_n_steps=1,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(Tiny(), loader(), loader())
    rank = os.environ.get("LOCAL_RANK", "0")
    with open(os.path.join(cfg.ckpt_root, f"rank{rank}.txt"), "w") as f:
        f.write(os.environ.get("TRACKING_RUN_ID", ""))

if __name__ == "__main__":
    main()
"""

CONFIG = """
ckpt_root: {root}
hydra:
  callbacks:
    mlflow:
      _target_: __pkg__.mlflow_hydra.MlflowCallback
      experiment: {experiment}
"""


def test_ddp_under_hydra_one_run(mlflow_env, tmp_path, model_module):
    shutil.copytree(TEMPLATES / "__pkg__", tmp_path / "__pkg__")
    name = f"ddp-{uuid.uuid4().hex[:6]}"
    root = tmp_path / "ckpt"
    root.mkdir()
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "config.yaml").write_text(CONFIG.format(root=root, experiment=name))
    env = {**os.environ, "RUNNER_GIT_SHA": "cafe"}
    out = subprocess.run(
        [sys.executable, "app.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    exp = MlflowClient().get_experiment_by_name(name)
    (run,) = MlflowClient().search_runs([exp.experiment_id])
    assert run.info.status == "FINISHED" and "train_loss" in run.data.metrics
    # rank 1 knew the run id and used the same checkpoint directory
    assert (root / "rank0.txt").read_text() == run.info.run_id
    assert (root / "rank1.txt").read_text() == run.info.run_id
    assert (root / run.info.run_id / "last.ckpt").is_file()
    assert [p.name for p in root.iterdir() if p.is_dir()] == [run.info.run_id]
