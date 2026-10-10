#!/usr/bin/env python3
"""Read an MLflow run: a sweep parent as a table of its children, any other
run as its tags, params, last metrics and artifacts; a metric's history; the
selected checkpoints, downloaded and checked against their sha256 tags.

    python3 show_run.py <run_id>                          # parent: children table
    python3 show_run.py <run_id> --metrics val_loss,acc --sort val_loss
    python3 show_run.py <run_id> --history val_loss       # step,value,timestamp CSV
    python3 show_run.py <run_id> --download ckpt/         # checkpoints + sha256 check
    python3 show_run.py <run_id> --json

Run ids come from find_runs.py. Needs mlflow-skinny: run it with the
project's environment (`uv run`) or `uv run --with mlflow-skinny python3
show_run.py ...`. Credentials come from ~/.mlflow/credentials, the server
from --tracking-uri, MLFLOW_TRACKING_URI or the project's mlflow.toml.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from _config import default_uri

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# tags worth showing first; the rest of the run's tags are MLflow's or the project's
REPRO_TAGS = (
    "git.sha",
    "git.dirty",
    "git.dirty.paths",
    "runner.task_id",
    "exp",
    "sweep.role",
    "mlflow.parentRunId",
    "hydra.job.num",
    "init_from.run_id",
)


def _q(v: str) -> str:
    return "'" + v.replace("'", "\\'") + "'"


def _time(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


def summary(client, run_id: str) -> dict:
    """A run's tags, params, last metric values, checkpoint digests and top-level artifacts."""
    r = client.get_run(run_id)
    tags = r.data.tags
    return {
        "run_id": run_id,
        "name": r.info.run_name,
        "experiment_id": r.info.experiment_id,
        "status": r.info.status,
        "start": _time(r.info.start_time),
        "end": _time(r.info.end_time),
        "tags": {k: tags[k] for k in REPRO_TAGS if k in tags},
        "params": dict(sorted(r.data.params.items())),
        "metrics": dict(sorted(r.data.metrics.items())),
        "sha256": {
            k[len("sha256.") :]: v for k, v in sorted(tags.items()) if k.startswith("sha256.")
        },
        "artifacts": [f.path + ("/" if f.is_dir else "") for f in client.list_artifacts(run_id)],
    }


def children(client, parent_run_id: str, metrics: list[str] | None = None, sort: str | None = None):
    """A sweep's children, one row each: job number, status, the params that differ
    between children, and the last value of each metric. Returns (rows, varying params)."""
    parent = client.get_run(parent_run_id)
    runs, token = [], None
    while True:
        page = client.search_runs(
            [parent.info.experiment_id],
            f"tags.`mlflow.parentRunId` = {_q(parent_run_id)}",
            max_results=1000,
            page_token=token,
        )
        runs += list(page)
        token = page.token
        if not token:
            break

    keys = set().union(*(r.data.params for r in runs)) if runs else set()
    varying = sorted(k for k in keys if len({r.data.params.get(k) for r in runs}) > 1)
    if metrics is None:
        metrics = sorted(set().union(*(r.data.metrics for r in runs))) if runs else []
    rows = []
    for r in runs:
        job = r.data.tags.get("hydra.job.num")
        rows.append(
            {
                "job": int(job) if job and job.isdigit() else job,
                "run_id": r.info.run_id,
                "status": r.info.status,
                "params": {k: r.data.params.get(k) for k in varying},
                "metrics": {m: r.data.metrics.get(m) for m in metrics},
            }
        )
    if sort:
        # runs without the metric go last
        rows.sort(key=lambda x: (x["metrics"].get(sort) is None, x["metrics"].get(sort) or 0))
    else:
        rows.sort(key=lambda x: (x["job"] is None, str(x["job"]).zfill(8)))
    return rows, varying


def history(client, run_id: str, metric: str) -> list[dict]:
    """Every logged value of a metric, by step."""
    h = client.get_metric_history(run_id, metric)
    return [
        {"step": m.step, "value": m.value, "timestamp": _time(m.timestamp)}
        for m in sorted(h, key=lambda m: (m.step, m.timestamp))
    ]


def download_checkpoints(client, run_id: str, dst: str | os.PathLike, artifact_path="checkpoints"):
    """Download an artifact directory and check each file against its
    `sha256.<artifact_path>/<name>` tag (set by tracking.log_checkpoint).
    Returns {local path: "ok" | "MISMATCH" | "no tag"}."""
    tags = client.get_run(run_id).data.tags
    local = Path(client.download_artifacts(run_id, artifact_path, str(dst)))
    files = [local] if local.is_file() else sorted(p for p in local.rglob("*") if p.is_file())
    out = {}
    for p in files:
        rel = artifact_path if p == local else f"{artifact_path}/{p.relative_to(local).as_posix()}"
        # the same key tracking.log_checkpoint writes
        want = tags.get("sha256." + re.sub(r"[^\w./ -]", "_", rel))
        if want is None:
            out[str(p)] = "no tag"
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(p)] = "ok" if h.hexdigest() == want else "MISMATCH"
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _table(header: list[str], rows: list[list[str]]) -> str:
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w[i]) for i, h in enumerate(header))]
    lines += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(line.rstrip() for line in lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_id")
    ap.add_argument(
        "--metrics", help="comma-separated metrics for the children table (default: all)"
    )
    ap.add_argument("--sort", metavar="METRIC", help="order the children by this metric, ascending")
    ap.add_argument("--history", metavar="METRIC", help="print every value of METRIC as CSV")
    ap.add_argument("--download", metavar="DIR", help="download checkpoints and check their sha256")
    ap.add_argument(
        "--artifact-path", default="checkpoints", help="for --download (default: checkpoints)"
    )
    ap.add_argument(
        "--tracking-uri",
        default=default_uri(),
        help="default: MLFLOW_TRACKING_URI, else tracking_uri in the project's mlflow.toml",
    )
    ap.add_argument("--json", action="store_true", help="print JSON")
    args = ap.parse_args()
    if not args.tracking_uri:
        ap.error("no server: pass --tracking-uri, set MLFLOW_TRACKING_URI or add mlflow.toml")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    if args.history:
        h = history(client, args.run_id, args.history)
        if args.json:
            print(json.dumps(h, indent=2))
        else:
            w = csv.DictWriter(sys.stdout, ["step", "value", "timestamp"], lineterminator="\n")
            w.writeheader()
            w.writerows(h)
        if not h:
            print(f"no values of {args.history!r} in run {args.run_id}", file=sys.stderr)
            return 1
        return 0

    if args.download:
        res = download_checkpoints(client, args.run_id, args.download, args.artifact_path)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            for p, s in res.items():
                print(f"{s:<8}  {p}")
        return 2 if "MISMATCH" in res.values() else 0

    s = summary(client, args.run_id)
    if s["tags"].get("sweep.role") == "parent":
        metrics = args.metrics.split(",") if args.metrics else None
        rows, varying = children(client, args.run_id, metrics, args.sort)
        if args.json:
            print(json.dumps({**s, "children": rows}, indent=2))
            return 0
        _print_summary(s, file=sys.stderr)
        ms = list(rows[0]["metrics"]) if rows else (metrics or [])
        print(
            _table(
                ["job", "status", "run_id", *varying, *ms],
                [
                    [_fmt(r["job"]), r["status"], r["run_id"]]
                    + [_fmt(r["params"][k]) for k in varying]
                    + [_fmt(r["metrics"][m]) for m in ms]
                    for r in rows
                ],
            )
        )
        return 0

    if args.json:
        print(json.dumps(s, indent=2))
    else:
        _print_summary(s, file=sys.stdout)
    return 0


def _print_summary(s: dict, file) -> None:
    p = functools.partial(print, file=file)
    p(f"{s['run_id']}  {s['status']}  {s['name']}  experiment {s['experiment_id']}")
    p(f"  start {s['start']}  end {s['end'] or '-'}")
    for k, v in s["tags"].items():
        p(f"  {k} = {v}")
    if s["tags"].get("sweep.role") == "parent":
        return
    if s["params"]:
        p("params:")
        for k, v in s["params"].items():
            p(f"  {k} = {v}")
    if s["metrics"]:
        p("metrics (last value):")
        for k, v in s["metrics"].items():
            p(f"  {k} = {_fmt(v)}")
    if s["sha256"]:
        p("sha256:")
        for k, v in s["sha256"].items():
            p(f"  {k}  {v}")
    if s["artifacts"]:
        p("artifacts: " + "  ".join(s["artifacts"]))


if __name__ == "__main__":
    sys.exit(main())
