import os

from Evaluation._config import AMOUNT_OF_CLASSIFIERS
from Evaluation.benchmark_protokoll import run_bench_mogon

if __name__ == "__main__":
    job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print(f"{job_id} / 152")
    string_id = job_id // AMOUNT_OF_CLASSIFIERS
    classifier_id = job_id % AMOUNT_OF_CLASSIFIERS
    print(f"{string_id}, {classifier_id}")
    # run_bench_mogon(string_id, classifier_id)
