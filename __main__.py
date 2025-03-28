import asyncio
import logging
from pathlib import Path

from Evaluation.EvaluationFunctions import clean_incomplete_directories, find_incomplete_directories, rename_folders
from Evaluation.benchmark_protokoll import bench_async, run_bench

# lr=0.0408738, weight-lr=0.00139971, delta=2.06634e-07, mci=8.66358e-06, (4, 'global_grace'), electricity, NLLLoss: 84.815%

if __name__ == "__main__":
    logging.basicConfig(filename=Path('bench_async.log').absolute().as_posix(), level=logging.INFO)
    run_id = 99
    find_incomplete_directories(run_id)
    rename_folders(run_id)
    clean_incomplete_directories(run_id)
    asyncio.run(bench_async())
    # run_bench()