from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.retail_stock.data_model import StockRetailDB
from tau2.domains.retail_stock.tools import RetailStockTools
from tau2.domains.retail_stock.utils import (
    RETAIL_STOCK_DB_PATH,
    RETAIL_STOCK_POLICY_PATH,
    RETAIL_STOCK_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[StockRetailDB] = None, solo_mode: bool = False
) -> Environment:
    """Build the isolated fixed-stock retail environment."""

    if solo_mode:
        raise ValueError("Retail stock domain does not support solo mode")
    if db is None:
        db = StockRetailDB.load(RETAIL_STOCK_DB_PATH)
    with open(RETAIL_STOCK_POLICY_PATH) as fp:
        policy = fp.read()
    return Environment(
        domain_name="retail_stock", policy=policy, tools=RetailStockTools(db)
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load the single fixed-stock task."""

    tasks = [
        Task.model_validate(task) for task in load_file(RETAIL_STOCK_TASK_SET_PATH)
    ]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    """Load task splits for the fixed-stock task set."""

    split_path = (
        Path(RETAIL_STOCK_TASK_SET_PATH).parent
        / f"split_{Path(RETAIL_STOCK_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_path)
