"""Chaotic systems versus the model zoo: classical, nearest-neighbour, deep and foundation.

Three generated chaotic series and one real one meet a cross-section of the
registry: numpy baselines, lag-window learners, empirical dynamic modelling,
a deep forecaster trained per origin and a time-series foundation model used
zero-shot. Models whose library is missing are skipped, so the example runs
on a plain install and grows with the extras you add.

Run:  python examples/02_chaos_vs_foundation.py --origins 5
      python examples/02_chaos_vs_foundation.py --fast       # 3 training epochs for deep models
"""

import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origins", type=int, default=5)
    p.add_argument("--out", default="results/chaos-vs-foundation")
    p.add_argument("--fast", action="store_true",
                   help="few training epochs for the wrappers that honour FORECAST_ARENA_FAST_MODE")
    p.add_argument("--no-network", action="store_true", help="use the bundled yearly sunspots")
    args = p.parse_args()
    if args.fast:
        os.environ["FORECAST_ARENA_FAST_MODE"] = "1"

    from forecast_arena import Protocol, per_task_table, run_arena, summarise, to_markdown
    from forecast_arena.runner import available_models

    candidates = [
        "Naive", "Drift",                                   # floor
        "Ridge", "KNN", "ExtraTrees",                       # lag-window learners
        "SugiharaSimplex", "SugiharaSMap", "GibsonLegendre",  # state-space reconstruction
        "StatsForecastAutoARIMA", "StatsForecastAutoTheta",   # classical automatic
        "NBEATS",                                           # deep, trained per origin (darts)
        "Chronos2",                                         # foundation model, zero-shot (chronos-bolt)
    ]
    available = set(available_models())
    models = [m for m in candidates if m in available]
    skipped = [m for m in candidates if m not in available]
    if skipped:
        print(f"Skipping (library not installed): {skipped}", file=sys.stderr)

    tasks = ["mackey-glass", "thomas-x", "henon-x",
             "sunspots-yearly" if args.no_network else "sunspots-monthly"]

    df = run_arena(tasks, models, protocol=Protocol(n_origins=args.origins), output_dir=args.out,
                   model_kwargs={"Chronos2": {"model_size": "tiny"}})

    table = per_task_table(df)
    for task in tasks:
        sub = table[table["task"] == task][["rank", "model", "mae", "mase", "n_errors", "fit_time_s"]]
        print(f"\n### {task}\n")
        print(to_markdown(sub))
    print("\n### Across tasks (mean rank)\n")
    print(to_markdown(summarise(table)))


if __name__ == "__main__":
    main()
