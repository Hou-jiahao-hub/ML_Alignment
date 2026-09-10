import importlib

import numpy as np
import pytest


def test_event_key_hash_is_order_and_duplicate_independent():
    m = importlib.import_module("scripts.event_identity")

    assert m.event_key_hash([2, 1, 1, 3]) == m.event_key_hash([3, 2, 1])
    assert m.event_key_hash([1, 2, 3, 4]) != m.event_key_hash([1, 2, 3])


@pytest.mark.parametrize(
    "event_ids",
    [
        [True, False],
        [1.25],
        [np.nan],
        [np.inf],
        [-1],
        [np.iinfo(np.int64).max + 1],
    ],
)
def test_event_key_hash_rejects_values_that_are_not_non_negative_int64(event_ids):
    """Invalid IDs must not be silently coerced into another event universe."""
    m = importlib.import_module("scripts.event_identity")

    with pytest.raises(ValueError, match="event ids"):
        m.event_key_hash(event_ids)
