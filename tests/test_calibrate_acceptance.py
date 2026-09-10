import importlib
import json

import pandas as pd


def test_cli_writes_schema_versioned_frozen_baseline(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.calibrate_acceptance")
    rows = []
    for event_nr in range(100):
        rows.append(
            {
                "event_nr": event_nr,
                "volume_id": 17,
                "layer_id": 8,
                "l_x_hit": 1.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            }
        )
    monkeypatch.setattr(
        m,
        "load_reconstruction_frame",
        lambda _path, **_kwargs: pd.DataFrame(rows),
    )
    output = tmp_path / "baseline.json"

    status = m.main(
        [
            "--states",
            str(tmp_path / "nominal.root"),
            "--output",
            str(output),
            "--target-layers",
            "17:8",
            "--bootstrap-replicates",
            "10",
        ]
    )

    result = json.loads(output.read_text())
    accept = importlib.import_module("scripts.reconstruction_acceptance")
    assert status == 0
    assert result["schema_version"] == "3.0"
    assert result["selection_version"] == accept.SELECTION_VERSION
    assert len(result["event_identity"]["event_key_sha256"]) == 64
    assert result["source"]["states"] == str(tmp_path / "nominal.root")
    assert result["layers"]["17:8"]["LOC0"]["threshold"] == 1.0
