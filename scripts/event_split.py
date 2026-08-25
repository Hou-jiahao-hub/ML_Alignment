"""Stable event partitions shared by alignment and validation workflows."""

from __future__ import annotations

import numpy as np


DEFAULT_SPLIT_SEED = 6006
PARTITIONS = ("train", "calibration", "validation")


def event_buckets(event_ids, seed: int = DEFAULT_SPLIT_SEED) -> np.ndarray:
    values = np.asarray(event_ids)
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("event ids must be finite integers")
    integers = values.astype(np.int64, copy=False)
    if (integers < 0).any():
        raise ValueError("event ids must be non-negative")
    return np.mod(integers * 37 + int(seed), 100)


def event_partition(event_ids, seed: int = DEFAULT_SPLIT_SEED) -> np.ndarray:
    buckets = event_buckets(event_ids, seed)
    return np.where(
        buckets < 80,
        "train",
        np.where(buckets < 90, "calibration", "validation"),
    )


def event_partition_mask(
    event_ids, partition: str, seed: int = DEFAULT_SPLIT_SEED
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise ValueError(
            f"Unknown event partition '{partition}'; expected one of {PARTITIONS}."
        )
    return event_partition(event_ids, seed) == partition
