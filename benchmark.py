import logging
import os
from datetime import datetime
from pathlib import Path

from Evaluation._config import AMOUNT_OF_CLASSIFIERS, AMOUNT_HYPERPARAMETER_TESTS, AMOUNT_OF_STRINGS, RESULTS_DIR_PATH, \
    STANDARD_RUN_ID
from Evaluation.benchmark_protokoll import bench_one_feature, bench_one_hyperparameter_isolated, bench_stable, \
    bench_unstable

if __name__ == "__main__":
    logging.basicConfig(filename=Path("mogon_run.log").absolute().as_posix(), level=logging.INFO)
    (RESULTS_DIR_PATH/ f"runID={STANDARD_RUN_ID}").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"benchmark_mogon")
    job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    biggest_run_index = AMOUNT_HYPERPARAMETER_TESTS + (AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS) + 2 - 1
    run_name = f'{job_id}/{biggest_run_index}'
    logger.info(f"starting: {run_name}: {datetime.now()}")
    if job_id < AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS:
        logger.info(f'bench one feature: {run_name}')
        bench_one_feature(job_id, run_name)
    elif job_id - AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS < AMOUNT_HYPERPARAMETER_TESTS:
        logger.info(f'bench one hyperparameter: {run_name}')
        bench_one_hyperparameter_isolated(job_id, run_name)
    elif job_id - (AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS + AMOUNT_HYPERPARAMETER_TESTS) == 0:
        logger.info(f'bench stable: {run_name}')
        bench_stable(job_id, run_name)
    elif job_id - (AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS + AMOUNT_HYPERPARAMETER_TESTS) == 1:
        logger.info(f'bench unstable: {run_name}')
        bench_unstable(job_id, run_name)
    else:
        logger.info(f"unexpected job id: {job_id}, run name: {run_name}")
