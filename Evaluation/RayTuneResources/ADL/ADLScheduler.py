from ray.tune.schedulers import ASHAScheduler

from Evaluation._config import MAX_INSTANCES, MIN_INSTANCES

ADLScheduler = ASHAScheduler(
    time_attr='instances_seen',
    metric="score",
    mode="max",
    max_t=MAX_INSTANCES,
    grace_period=MIN_INSTANCES
)
