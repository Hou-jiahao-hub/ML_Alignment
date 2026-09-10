"""Event-set identity hashing shared by alignment and acceptance workflows."""

from __future__ import annotations

import hashlib

import numpy as np


def event_key_hash(event_ids) -> str:
    """Identity of a set of event ids, independent of order or duplicates.

    Binds a frozen acceptance profile to the event IDs present in the input
    trackstates it was calibrated on: two different observed event sets must
    not compare equal.
    """
    values = np.asarray(event_ids)
    if np.issubdtype(values.dtype, np.bool_):
        raise ValueError("event ids must be non-negative int64 values")
    if np.issubdtype(values.dtype, np.signedinteger):
        if (values < 0).any():
            raise ValueError("event ids must be non-negative int64 values")
        integers = values.astype(np.int64, copy=False)
    elif np.issubdtype(values.dtype, np.unsignedinteger):
        if (values > np.iinfo(np.int64).max).any():
            raise ValueError("event ids must fit in int64")
        integers = values.astype(np.int64, copy=False)
    elif np.issubdtype(values.dtype, np.floating):
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or (values < 0).any()
            or (values >= 2**63).any()
        ):
            raise ValueError("event ids must be finite non-negative int64 values")
        integers = values.astype(np.int64)
    else:
        raise ValueError("event ids must be non-negative int64 values")
    unique_sorted = np.unique(integers)
    return hashlib.sha256(unique_sorted.astype("<i8").tobytes()).hexdigest()
