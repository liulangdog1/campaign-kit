"""Pluggable batch execution: one interface, cluster and local backends."""

from campaign_kit.scheduler.base import (
    BatchScheduler,
    JobLedger,
    JobState,
    RetryPolicy,
    SchedulerConfig,
    SchedulerError,
)
from campaign_kit.scheduler.local import LocalScheduler
from campaign_kit.scheduler.slurm import SlurmScheduler

__all__ = [
    "BatchScheduler",
    "JobLedger",
    "JobState",
    "LocalScheduler",
    "RetryPolicy",
    "SchedulerConfig",
    "SchedulerError",
    "SlurmScheduler",
]
