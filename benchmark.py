import logging
import os
from pathlib import Path

from Evaluation._config import AMOUNT_OF_CLASSIFIERS, STANDARD_RUN_ID

if __name__ == "__main__":
    logging.basicConfig(filename=Path("mogon_run.log").absolute().as_posix(), level=logging.INFO)
    logger = logging.getLogger(f"logger_runID={STANDARD_RUN_ID}")
    job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    logger.info(f"{job_id} / 10")
    print(f"{job_id} / 10")
    string_id = job_id // AMOUNT_OF_CLASSIFIERS
    classifier_id = job_id % AMOUNT_OF_CLASSIFIERS
    print(f"{string_id}, {classifier_id}")
    # test_one_classifier_one_stream_on_standard_config(string_id, classifier_id)
