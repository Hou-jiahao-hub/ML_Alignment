import importlib

import numpy as np
import pytest


def test_event_partition_uses_stable_80_10_10_buckets():
    m = importlib.import_module("scripts.event_split")

    assert m.event_partition(np.asarray([0, 2, 5]), seed=6006).tolist() == [
        "train",
        "calibration",
        "validation",
    ]


def test_event_partition_is_complete_and_rejects_unknown_partition():
    m = importlib.import_module("scripts.event_split")
    event_ids = np.arange(100)

    masks = [m.event_partition_mask(event_ids, name, seed=6006) for name in m.PARTITIONS]

    assert np.sum(masks, axis=0).tolist() == [1] * 100
    assert [int(mask.sum()) for mask in masks] == [80, 10, 10]
    with pytest.raises(ValueError, match="Unknown event partition"):
        m.event_partition_mask(event_ids, "test", seed=6006)

