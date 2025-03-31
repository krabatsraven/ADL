import asyncio
import logging
from pathlib import Path

from Evaluation.EvaluationFunctions import clean_incomplete_directories, find_incomplete_directories, rename_folders
from Evaluation._config import STANDARD_RUN_ID
from Evaluation.benchmark_protokoll import bench_async, run_bench


if __name__ == "__main__":
    logging.basicConfig(filename=Path('bench.log').absolute().as_posix(), level=logging.INFO)
    run_id = STANDARD_RUN_ID
    find_incomplete_directories(run_id)
    rename_folders(run_id)
    clean_incomplete_directories(run_id)
    asyncio.run(bench_async())
    run_bench()