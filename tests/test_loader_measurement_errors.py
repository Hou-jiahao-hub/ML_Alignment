"""A5: measurement-error branches are required, and length mismatches must be loud."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
import uproot

from mlalign.root_inputs import load_variable_information


pytestmark = pytest.mark.root_io




# Mirrors tests/test_joint_alignment.py::test_root_loader_exposes_raw_joint_...
# NOTE: the NaN sits in l_y_hit, not l_x_hit.  The measurement count is derived
# from ~isnan(l_x_hit), so this entry has THREE measurements, not two.
STATE_VALUES = {
    "volume_id": [16, 10, 10],
    "layer_id": [6, 2, 1],
    "module_id": [3, 2, 1],
    "l_x_hit": [0.3, 0.2, 0.1],
    "l_y_hit": [0.4, np.nan, 0.2],
    "eLOC0_smt": [0.0, 0.0, 0.0],
    "eLOC1_smt": [0.0, 0.0, 0.0],
    "g_x_smt": [30.0, 20.0, 10.0],
    "g_y_smt": [0.0, 0.0, 0.0],
    "g_z_smt": [0.0, 0.0, 0.0],
    "px_smt": [1.0, 1.0, 1.0],
    "py_smt": [0.0, 0.0, 0.0],
    "pz_smt": [0.0, 0.0, 0.0],
    "ePHI_smt": [0.0, 0.0, 0.0],
    "eTHETA_smt": [1.0, 1.0, 1.0],
    "eQOP_smt": [0.001, 0.001, 0.001],
}
N_MEASUREMENTS = 3
ID_MAP = {(16, 6, 3): 2, (10, 2, 2): 1, (10, 1, 1): 0}


def write_root(tmp_path, err_x, err_y):
    """``err_x``/``err_y`` of None omit the branch entirely."""
    summary_path = tmp_path / "summary.root"
    states_path = tmp_path / "states.root"
    with uproot.recreate(summary_path) as root_file:
        root_file["tracksummary"] = {
            "event_nr": np.array([1], dtype=np.int64),
            "track_nr": np.array([7], dtype=np.int64),
            "chi2Sum": np.array([2.0]),
            "NDF": np.array([1.0]),
        }
    with uproot.recreate(states_path) as root_file:
        tree = {name: ak.Array([values]) for name, values in STATE_VALUES.items()}
        tree.update({
            "event_nr": np.array([1], dtype=np.int64),
            "track_nr": np.array([7], dtype=np.int64),
        })
        if err_x is not None:
            tree["err_x_hit"] = ak.Array([err_x])
        if err_y is not None:
            tree["err_y_hit"] = ak.Array([err_y])
        root_file["trackstates"] = tree
    return summary_path, states_path


GOOD_X = [0.1, 0.1, 0.1]
GOOD_Y = [0.2, np.nan, 0.2]


def test_loader_accepts_matching_measurement_error_lengths(tmp_path):
    summary_path, states_path = write_root(tmp_path, GOOD_X, GOOD_Y)

    inputs = load_variable_information(summary_path, states_path, ID_MAP)

    assert len(inputs.hit_sensor_ids) == 3


@pytest.mark.parametrize("branch", ["err_x_hit", "err_y_hit"])
def test_loader_raises_when_measurement_error_branch_missing(tmp_path, branch):
    err_x = None if branch == "err_x_hit" else GOOD_X
    err_y = None if branch == "err_y_hit" else GOOD_Y
    summary_path, states_path = write_root(tmp_path, err_x, err_y)

    with pytest.raises(ValueError) as excinfo:
        load_variable_information(summary_path, states_path, ID_MAP)

    message = str(excinfo.value)
    assert "err_x_hit" in message and "err_y_hit" in message
    assert "posterior variance" not in message


@pytest.mark.parametrize("branch", ["err_x_hit", "err_y_hit"])
def test_loader_raises_with_entry_and_length_on_mismatch(tmp_path, branch):
    """The top-level message must carry the cause's entry/length detail."""
    err_x = [0.1] if branch == "err_x_hit" else GOOD_X
    err_y = [0.2] if branch == "err_y_hit" else GOOD_Y
    summary_path, states_path = write_root(tmp_path, err_x, err_y)

    with pytest.raises(ValueError) as excinfo:
        load_variable_information(summary_path, states_path, ID_MAP)

    message = str(excinfo.value)
    assert branch in message
    assert "entry 0" in message
    assert "1" in message and str(N_MEASUREMENTS) in message


def test_loader_does_not_claim_a_posterior_variance_fallback(tmp_path, capsys):
    """The old log line advertised a fallback that never existed."""
    summary_path, states_path = write_root(tmp_path, GOOD_X, GOOD_Y)

    load_variable_information(summary_path, states_path, ID_MAP)

    assert "posterior variance" not in capsys.readouterr().out
