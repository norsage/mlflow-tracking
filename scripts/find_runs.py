#!/usr/bin/env python3
"""Find the top-level MLflow runs (sweep parents and single runs) of a task,
a commit or a research-log experiment, and print the arguments for
research-log's record_run.py.

    python3 find_runs.py --task-id 12345
    python3 find_runs.py --sha 3f2a9c1 --tracking-uri https://mlflow.example.org
    python3 find_runs.py --exp E-0007.k2f --json

Needs mlflow-skinny: run it with the project's environment (`uv run`) or
`uv run --with mlflow-skinny python3 find_runs.py ...`. Credentials come
from ~/.mlflow/credentials, the server from --tracking-uri,
MLFLOW_TRACKING_URI or the project's mlflow.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime

from _config import default_uri

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _q(v: str) -> str:
    return "'" + v.replace("'", "\\'") + "'"


def find(client, *, task_id=None, sha=None, exp=None, experiment=None):
    """Top-level runs matching one filter, oldest first, each with its children."""
    from mlflow.entities import ViewType

    if task_id:
        cond = f"tags.`runner.task_id` = {_q(task_id)}"
    elif exp:
        cond = f"tags.`exp` = {_q(exp)}"
    elif sha:
        # a prefix is enough, as with git
        cond = f"tags.`git.sha` LIKE {_q(sha + '%')}"
    else:
        raise ValueError("one of task_id, sha, exp is required")

    if experiment:
        e = client.get_experiment_by_name(experiment)
        if e is None:
            raise SystemExit(f"no experiment named {experiment!r}")
        exp_ids = [e.experiment_id]
    else:
        exp_ids = [e.experiment_id for e in client.search_experiments(view_type=ViewType.ALL)]

    found = []
    for role in ("parent", "single"):  # MLflow has no IN over tags
        flt = f"{cond} and tags.`sweep.role` = '{role}'"
        found += _search_all(client, exp_ids, flt)
    found.sort(key=lambda r: r.info.start_time)

    out = []
    for r in found:
        children = []
        if r.data.tags.get("sweep.role") == "parent":
            children = _search_all(
                client, [r.info.experiment_id], f"tags.`mlflow.parentRunId` = {_q(r.info.run_id)}"
            )
        out.append(
            {
                "run_id": r.info.run_id,
                "experiment_id": r.info.experiment_id,
                "name": r.info.run_name,
                "status": r.info.status,
                "role": r.data.tags.get("sweep.role"),
                "start": datetime.fromtimestamp(r.info.start_time / 1000, UTC).isoformat(),
                "sha": r.data.tags.get("git.sha"),
                "dirty": r.data.tags.get("git.dirty"),
                "task_id": r.data.tags.get("runner.task_id"),
                "exp": r.data.tags.get("exp"),
                "children": dict(Counter(c.info.status for c in children)),
            }
        )
    return out


def _search_all(client, exp_ids, flt):
    runs, token = [], None
    for i in range(0, len(exp_ids), 100):  # search_runs takes at most 100 experiments
        chunk = exp_ids[i : i + 100]
        while True:
            page = client.search_runs(chunk, flt, max_results=1000, page_token=token)
            runs += list(page)
            token = page.token
            if not token:
                break
    return runs


def record_run_args(runs) -> str:
    """`--tracker mlflow --run-id ... --commit ... --date ...` for record_run.py."""
    parts = ["--tracker mlflow"] + [f"--run-id {r['run_id']}" for r in runs]
    shas = {r["sha"] for r in runs if r["sha"]}
    if len(shas) == 1:
        parts.append(f"--commit {shas.pop()}")
    if runs:
        parts.append(f"--date {min(r['start'] for r in runs)[:10]}")
    if any(r["dirty"] == "true" for r in runs):
        parts.append("--allow-dirty")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-id", help="the transport's task id (tag runner.task_id)")
    g.add_argument("--sha", help="a commit or its prefix (tag git.sha)")
    g.add_argument("--exp", help="a research-log experiment id (tag exp)")
    ap.add_argument("--experiment", help="restrict to this MLflow experiment name")
    ap.add_argument(
        "--tracking-uri",
        default=default_uri(),
        help="default: MLFLOW_TRACKING_URI, else tracking_uri in the project's mlflow.toml",
    )
    ap.add_argument("--json", action="store_true", help="print the runs as JSON")
    args = ap.parse_args()
    if not args.tracking_uri:
        ap.error("no server: pass --tracking-uri, set MLFLOW_TRACKING_URI or add mlflow.toml")

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    runs = find(
        MlflowClient(), task_id=args.task_id, sha=args.sha, exp=args.exp, experiment=args.experiment
    )
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0 if runs else 1
    if not runs:
        print(
            "no runs found. If the task ran, its run was not created: see the task log "
            "for 'MLflow run not created'.",
            file=sys.stderr,
        )
        return 1
    for r in runs:
        kids = ", ".join(f"{n} {s}" for s, n in sorted(r["children"].items()))
        print(
            f"{r['run_id']}  {r['role']:<6}  {r['status']:<8}  {r['start'][:16]}  "
            f"{r['name']}" + (f"  children: {kids}" if kids else ""),
            file=sys.stderr,
        )
    if any(r["status"] != "FINISHED" for r in runs):
        print("note: not every run finished", file=sys.stderr)
    print(record_run_args(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
