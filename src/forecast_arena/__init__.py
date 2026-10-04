"""forecast-arena: one rolling-origin protocol for every forecaster."""

from ._version import __version__
from .forecasters import Forecaster, ModelCategory, ModelMetadata, ParamSpace, get_model, list_models, register_model
from .splits import describe_splits, generate_rolling_origins
from .tasks import TaskSpec, get_task, list_tasks, load_series, register_csv_task, register_task
from .arena import Protocol, compute_metrics, run_arena
from .leaderboard import load_records, per_task_table, summarise, task_leaderboard, to_markdown
from .runner import run_models, run_single_model

__all__ = [
    "__version__",
    "Forecaster", "ModelCategory", "ModelMetadata", "ParamSpace",
    "get_model", "list_models", "register_model",
    "describe_splits", "generate_rolling_origins",
    "TaskSpec", "get_task", "list_tasks", "load_series", "register_csv_task", "register_task",
    "Protocol", "compute_metrics", "run_arena",
    "load_records", "per_task_table", "summarise", "task_leaderboard", "to_markdown",
    "run_models", "run_single_model",
]
