"""Command-line entry point: ``forecast-arena tasks | models | run | leaderboard``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _cmd_tasks(args) -> int:
    from .tasks import tasks_table
    table = tasks_table()
    if args.offline:
        table = table[table["offline"]]
    cols = ["task", "domain", "freq", "seasonality", "offline", "license"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table[cols].to_string(index=False))
    return 0


def _cmd_models(args) -> int:
    from .forecasters import get_all_metadata
    from .runner import library_available
    rows = [{
        "model": m.name, "category": m.category.value, "library": m.library, "year": m.year,
        "zero_shot": m.zero_shot, "gpu": m.requires_gpu, "available": library_available(m.name),
    } for m in get_all_metadata()]
    df = pd.DataFrame(rows).sort_values(["category", "library", "model"])
    if args.library:
        df = df[df["library"] == args.library]
    if args.category:
        df = df[df["category"] == args.category]
    if args.available:
        df = df[df["available"]]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))
    n_avail = int(df["available"].sum())
    print(f"\n{len(df)} models registered; {n_avail} have their library installed here.")
    return 0


def _cmd_run(args) -> int:
    from .arena import Protocol, run_arena
    from .leaderboard import per_task_table, to_markdown
    from .runner import select_models

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = (select_models(library=args.library) if args.library
              else [m.strip() for m in (args.models or "").split(",") if m.strip()])
    if not models:
        print("Give --models A,B,C or --library <name>.", file=sys.stderr)
        return 2
    protocol = Protocol(n_origins=args.origins, min_train_frac=args.min_train,
                        max_train_frac=args.max_train, min_test_len=args.min_test, seed=args.seed)
    df = run_arena(tasks, models, protocol=protocol, output_dir=args.out, resume=not args.fresh,
                   verbose=not args.quiet, store_forecasts=not args.no_forecasts)
    table = per_task_table(df)
    for task in tasks:
        sub = table[table["task"] == task][["rank", "model", "mae", "rmse", "mase", "n_errors"]]
        print(f"\n### {task}\n")
        print(to_markdown(sub))
    return 0


def _cmd_leaderboard(args) -> int:
    from .leaderboard import load_records, per_task_table, summarise, to_markdown
    df = load_records(args.results)
    if df.empty:
        print(f"No records under {args.results}", file=sys.stderr)
        return 1
    table = per_task_table(df, metric=args.metric)
    if args.summary:
        try:
            print(to_markdown(summarise(table, rule=args.rule)))
        except NotImplementedError as e:
            print(str(e), file=sys.stderr)
            return 2
        return 0
    for task in sorted(table["task"].unique()):
        sub = table[table["task"] == task][["rank", "model", "mae", "rmse", "smape", "mase", "n_errors"]]
        print(f"\n### {task}\n")
        print(to_markdown(sub, max_rows=args.top))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forecast-arena",
                                description="One rolling-origin protocol for every forecaster.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tasks", help="list the task atlas")
    t.add_argument("--offline", action="store_true", help="only tasks that need no download")
    t.set_defaults(func=_cmd_tasks)

    m = sub.add_parser("models", help="list registered models importable here")
    m.add_argument("--library")
    m.add_argument("--category")
    m.add_argument("--available", action="store_true", help="only models whose library is installed")
    m.set_defaults(func=_cmd_models)

    r = sub.add_parser("run", help="evaluate models on tasks under the protocol")
    r.add_argument("--tasks", required=True, help="comma-separated task ids")
    r.add_argument("--models", help="comma-separated model names")
    r.add_argument("--library", help="run every registered model from one library")
    r.add_argument("--origins", type=int, default=15)
    r.add_argument("--min-train", type=float, default=0.5)
    r.add_argument("--max-train", type=float, default=0.85)
    r.add_argument("--min-test", type=int, default=12)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results")
    r.add_argument("--fresh", action="store_true", help="ignore records already on disk")
    r.add_argument("--no-forecasts", action="store_true", help="do not store forecast vectors")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)

    lb = sub.add_parser("leaderboard", help="rank models from saved records")
    lb.add_argument("results", nargs="?", default="results")
    lb.add_argument("--metric", default="mae", choices=["mae", "rmse", "smape", "mase"])
    lb.add_argument("--summary", action="store_true", help="one row per model across tasks")
    lb.add_argument("--rule", default="mean_rank", choices=["mean_rank", "podium", "custom"])
    lb.add_argument("--top", type=int, default=None)
    lb.set_defaults(func=_cmd_leaderboard)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
