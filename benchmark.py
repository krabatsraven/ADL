import logging
import os
from pathlib import Path

from Evaluation._config import AMOUNT_OF_CLASSIFIERS
from Evaluation.benchmark_protokoll import run_bench_mogon

if __name__ == "__main__":
    logging.basicConfig(filename=Path("mogon_benchmark.log").absolute().as_posix())
    logger = logging.getLogger(f"logger_runID={99}")
    job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    logger.info(f"{job_id} / 10")
    print(f"{job_id} / 10")
    string_id = job_id // AMOUNT_OF_CLASSIFIERS
    classifier_id = job_id % AMOUNT_OF_CLASSIFIERS
    print(f"{string_id}, {classifier_id}")
    run_bench_mogon(string_id, classifier_id)
