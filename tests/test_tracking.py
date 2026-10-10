import os
import subprocess
import uuid

import mlflow
import pytest
from conftest import free_port, start_server, stop_server
from mlflow.tracking import MlflowClient

from __pkg__ import tracking


def exp_name() -> str:
    return f"t-{uuid.uuid4().hex[:8]}"


def get(run_id):
    return MlflowClient().get_run(run_id)


def git_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_local_run_tags_from_git(mlflow_env, tmp_path):
    sha = git_repo(tmp_path)
    with tracking.start_run(experiment=exp_name(), params={"a": {"b": 1}, "c": "x"}) as run:
        assert run.active and mlflow.active_run().info.run_id == run.run_id
        mlflow.log_metric("loss", 0.5, step=1)
    r = get(run.run_id)
    assert r.info.status == "FINISHED"
    assert r.data.tags["git.sha"] == sha
    assert r.data.tags["git.dirty"] == "false"
    assert r.data.tags["sweep.role"] == "single"
    assert "runner.task_id" not in r.data.tags
    assert r.data.params == {"a.b": "1", "c": "x"}
    assert r.data.metrics["loss"] == 0.5


def test_dirty_tree(mlflow_env, tmp_path):
    git_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    with tracking.start_run(experiment=exp_name()) as run:
        pass
    assert get(run.run_id).data.tags["git.dirty"] == "true"
    assert get(run.run_id).data.tags["git.dirty.paths"] == "f.txt"


def test_untracked_code_is_dirty_markdown_and_ignored_are_not(mlflow_env, tmp_path):
    git_repo(tmp_path)
    (tmp_path / "notes.md").write_text("draft")
    (tmp_path / ".gitignore").write_text("out/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ignore"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x.bin").write_text("x")
    with tracking.start_run(experiment=exp_name()) as run:
        pass
    assert get(run.run_id).data.tags["git.dirty"] == "false"

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "new.py").write_text("print(1)")
    with tracking.start_run(experiment=exp_name()) as run:
        pass
    tags = get(run.run_id).data.tags
    assert tags["git.dirty"] == "true"
    assert tags["git.dirty.paths"] == "scripts/new.py"


def test_runner_env_tags(mlflow_env, monkeypatch):
    monkeypatch.setenv("RUNNER_GIT_SHA", "abc123")
    monkeypatch.setenv("RUNNER_TASK_ID", "task-7")
    monkeypatch.setenv("RUNNER_EXP", "E-0003.k2f")
    with tracking.start_run(experiment=exp_name(), run_name="r1") as run:
        pass
    tags = get(run.run_id).data.tags
    assert tags["git.sha"] == "abc123" and tags["git.dirty"] == "false"
    assert tags["runner.task_id"] == "task-7"
    assert tags["exp"] == "E-0003.k2f"
    assert tags["mlflow.runName"] == "r1"


def test_arguments_override_env(mlflow_env, monkeypatch):
    monkeypatch.setenv("RUNNER_TASK_ID", "env-task")
    with tracking.start_run(experiment=exp_name(), task_id="arg-task", sha="s1", exp="E-1") as run:
        pass
    tags = get(run.run_id).data.tags
    assert tags["runner.task_id"] == "arg-task" and tags["git.sha"] == "s1" and tags["exp"] == "E-1"


def test_exception_marks_failed(mlflow_env):
    with pytest.raises(RuntimeError), tracking.start_run(experiment=exp_name(), sha="s") as run:
        raise RuntimeError("boom")
    assert get(run.run_id).info.status == "FAILED"
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "var,value", [("RANK", "1"), ("LOCAL_RANK", "2"), ("NODE_RANK", "1"), ("SLURM_PROCID", "3")]
)
def test_nonzero_rank_does_not_write(mlflow_env, monkeypatch, var, value):
    name = exp_name()
    monkeypatch.setenv("TRACKING_RUN_ID", "from-rank-0")
    monkeypatch.setenv(var, value)
    with tracking.start_run(experiment=name, sha="s") as run:
        assert not run.active and run.run_id == "from-rank-0"
        mlflow.log_metric("loss", 1.0)  # would otherwise start a run of its own
        mlflow.set_tag("k", "v")
    assert mlflow.active_run() is None
    assert MlflowClient().get_experiment_by_name(name) is None


def test_rank_precedence(monkeypatch):
    """LOCAL_RANK before SLURM_PROCID: SLURM may be set while not managing the ranks."""
    for v in ("RANK", "LOCAL_RANK", "NODE_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(v, raising=False)
    assert tracking.rank_zero()
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert tracking.rank_zero()
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert not tracking.rank_zero()


def test_rank_zero_exports_run_id(mlflow_env, monkeypatch):
    import os

    monkeypatch.setenv("LOCAL_RANK", "0")
    with tracking.start_run(experiment=exp_name(), sha="s") as run:
        assert run.active
        assert os.environ["TRACKING_RUN_ID"] == run.run_id


def test_server_down_at_start(mlflow_env, caplog):
    dead = f"http://127.0.0.1:{free_port()}"
    with tracking.start_run(experiment=exp_name(), sha="s", tracking_uri=dead) as run:
        assert not run.active
        mlflow.log_metric("loss", 1.0)
        mlflow.log_param("p", 1)
    assert "not created" in caplog.text
    # the next run in the same process (a sweep job) is not muted
    mlflow.set_tracking_uri(mlflow_env)
    with tracking.start_run(experiment=exp_name(), sha="s") as run2:
        assert run2.active
        mlflow.log_metric("loss", 2.0)
    assert get(run2.run_id).data.metrics["loss"] == 2.0


def test_server_lost_mid_run(mlflow_env, tmp_path_factory, tmp_path, caplog):
    proc, uri = start_server(tmp_path_factory.mktemp("lost"))
    try:
        run = tracking.start_run(experiment=exp_name(), sha="s", tracking_uri=uri)
        assert run.active
        mlflow.log_metric("loss", 1.0, step=0)
        mlflow.flush_async_logging()
    finally:
        stop_server(proc)
    for i in range(1, 50):
        mlflow.log_metric("loss", float(i), step=i)
    mlflow.set_tag("after", "outage")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"w")
    tracking.log_checkpoint(ckpt)
    tracking.end_run()
    assert mlflow.active_run() is None
    assert "failed" in caplog.text


def _failed_run(name, **kw):
    run = tracking.start_run(experiment=name, sha="s", **kw)
    mlflow.log_metric("step", 1)
    tracking.end_run("FAILED")
    return run.run_id


def test_resume_by_task_id(mlflow_env, monkeypatch):
    name = exp_name()
    monkeypatch.setenv("RUNNER_TASK_ID", "t-resume")
    first = _failed_run(name)
    with tracking.start_run(experiment=name, sha="s", params={"lr": 0.1}) as run:
        assert run.run_id == first
        mlflow.log_metric("step", 2)
    assert get(first).info.status == "FINISHED"
    # a finished run is not reopened
    with tracking.start_run(experiment=name, sha="s") as run3:
        assert run3.run_id != first


def test_resume_matches_extra_tags(mlflow_env, monkeypatch):
    name = exp_name()
    monkeypatch.setenv("RUNNER_TASK_ID", "t-match")
    a = _failed_run(name, tags={"hydra.job.num": 0}, resume_match={"hydra.job.num": 0})
    with tracking.start_run(
        experiment=name, sha="s", tags={"hydra.job.num": 1}, resume_match={"hydra.job.num": 1}
    ) as run:
        assert run.run_id != a
    with tracking.start_run(
        experiment=name, sha="s", tags={"hydra.job.num": 0}, resume_match={"hydra.job.num": 0}
    ) as run:
        assert run.run_id == a


def test_local_run_not_resumed_implicitly(mlflow_env):
    name = exp_name()
    first = _failed_run(name)
    with tracking.start_run(experiment=name, sha="s") as run:
        assert run.run_id != first


def test_resume_explicit(mlflow_env, monkeypatch):
    name = exp_name()
    first = _failed_run(name)
    monkeypatch.setenv("RUNNER_RESUME_RUN_ID", first)
    with tracking.start_run(experiment=name, sha="s") as run:
        assert run.run_id == first
    assert get(first).info.status == "FINISHED"


def test_resume_with_changed_param_does_not_fail(mlflow_env, caplog):
    name = exp_name()
    run = tracking.start_run(experiment=name, sha="s", params={"lr": 0.1})
    tracking.end_run("FAILED")
    with tracking.start_run(
        experiment=name, sha="s", resume_run_id=run.run_id, params={"lr": 0.2, "new": 1}
    ):
        pass
    mlflow.flush_async_logging()
    params = get(run.run_id).data.params
    assert params == {"lr": "0.1", "new": "1"}
    assert "immutable" in caplog.text


def test_log_checkpoint(mlflow_env, tmp_path):
    import hashlib

    ckpt = tmp_path / "epoch=3-step=10.ckpt"
    ckpt.write_bytes(b"weights" * 1000)
    with tracking.start_run(experiment=exp_name(), sha="s") as run:
        digest = tracking.log_checkpoint(ckpt)
    assert digest == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    r = get(run.run_id)
    assert r.data.tags["sha256.checkpoints/epoch_3-step_10.ckpt"] == digest
    files = [f.path for f in MlflowClient().list_artifacts(run.run_id, "checkpoints")]
    assert files == ["checkpoints/epoch=3-step=10.ckpt"]


def test_checkpoint_dir_and_prune(tmp_path):
    import os
    import time

    d = tracking.checkpoint_dir(tmp_path, "run123")
    assert d == tmp_path / "run123" and d.is_dir()
    paths = []
    for i in range(5):
        p = d / f"step{i}.ckpt"
        p.write_text(str(i))
        os.utime(p, (time.time() + i, time.time() + i))
        paths.append(p)
    removed = tracking.prune_checkpoints(d, "*.ckpt", keep_last=2, keep=[paths[0]])
    assert sorted(p.name for p in removed) == ["step1.ckpt", "step2.ckpt"]
    assert sorted(p.name for p in d.iterdir()) == ["step0.ckpt", "step3.ckpt", "step4.ckpt"]


def test_parent_not_activated(mlflow_env):
    parent = tracking.start_run(experiment=exp_name(), sha="s", role="parent", activate=False)
    assert parent.active and mlflow.active_run() is None
    tracking.terminate(parent.run_id)
    assert get(parent.run_id).info.status == "FINISHED"


def test_bad_role():
    with pytest.raises(ValueError):
        tracking.start_run(role="child-ish")


def test_resume_from_another_commit_starts_new_run(mlflow_env, monkeypatch):
    name = exp_name()
    monkeypatch.setenv("RUNNER_TASK_ID", "t-commit")
    first = _failed_run(name)  # sha "s"
    with tracking.start_run(experiment=name, sha="t") as run:
        assert run.run_id != first and not run.resumed and run.init_from == first
        assert os.environ["TRACKING_INIT_FROM"] == first
    assert get(run.run_id).data.tags["init_from.run_id"] == first
    assert get(first).info.status == "FAILED"  # the old run is left as it was
    assert "TRACKING_INIT_FROM" not in os.environ


def test_explicit_resume_from_another_commit(mlflow_env, monkeypatch):
    name = exp_name()
    first = _failed_run(name)
    monkeypatch.setenv("RUNNER_RESUME_RUN_ID", first)
    with tracking.start_run(experiment=name, sha="t") as run:
        assert run.run_id != first and run.init_from == first


def test_resume_same_commit_prefix(mlflow_env):
    name = exp_name()
    run = tracking.start_run(experiment=name, sha="abcdef0123")
    tracking.end_run("FAILED")
    with tracking.start_run(experiment=name, sha="abcdef0", resume_run_id=run.run_id) as again:
        assert again.run_id == run.run_id and again.resumed and again.init_from is None


def test_init_from_env(mlflow_env, monkeypatch):
    name = exp_name()
    first = _failed_run(name)
    monkeypatch.setenv("RUNNER_INIT_FROM_RUN_ID", first)
    with tracking.start_run(experiment=name, sha="s") as run:
        assert run.run_id != first and run.init_from == first


def test_checkpoint_root_is_shared(tmp_path):
    import stat

    root = tmp_path / "volume" / "ckpt"
    d = tracking.checkpoint_dir(root, "run1", shared=True)
    for p in (tmp_path / "volume", root):
        assert stat.S_IMODE(p.stat().st_mode) == 0o1777
    assert stat.S_IMODE(d.stat().st_mode) & 0o1777 != 0o1777  # the run's own


def test_checkpoint_root_private_by_default(tmp_path, monkeypatch):
    import stat

    monkeypatch.chdir(tmp_path)  # no mlflow.toml here or above
    root = tmp_path / "volume" / "ckpt"
    tracking.checkpoint_dir(root, "run1")
    for p in (tmp_path / "volume", root):
        assert not stat.S_IMODE(p.stat().st_mode) & (stat.S_IWOTH | stat.S_ISVTX)


def test_shared_checkpoints_from_mlflow_toml(tmp_path, monkeypatch):
    import stat

    (tmp_path / "mlflow.toml").write_text("shared_checkpoints = true\n")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "ckpt"
    tracking.checkpoint_dir(root, "run1")
    assert stat.S_IMODE(root.stat().st_mode) == 0o1777


def test_checkpoint_dir_of_another_user(tmp_path):
    d = tracking.checkpoint_dir(tmp_path, "theirs")
    d.chmod(0o555)  # as if another user's
    try:
        with pytest.raises(PermissionError, match="RUNNER_INIT_FROM_RUN_ID=theirs"):
            tracking.checkpoint_dir(tmp_path, "theirs")
    finally:
        d.chmod(0o755)


def test_init_checkpoint_dir(tmp_path):
    assert tracking.init_checkpoint_dir(tmp_path, tracking.Run("new", True)) is None
    old = tracking.checkpoint_dir(tmp_path, "old")
    run = tracking.Run("new", True, init_from="old")
    assert tracking.init_checkpoint_dir(tmp_path, run) == old


def test_prune_skips_what_it_cannot_remove(tmp_path):
    import time

    d = tracking.checkpoint_dir(tmp_path, "run")
    for i in range(3):
        p = d / f"step{i}.ckpt"
        p.write_text(str(i))
        os.utime(p, (time.time() + i, time.time() + i))
    d.chmod(0o555)
    try:
        assert tracking.prune_checkpoints(d, "*.ckpt", keep_last=1) == []
    finally:
        d.chmod(0o755)
