from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


TARGET_LAYERS = ((17, 8), (24, 4))
ACTIVE_DOF = (0, 1, 5)


def _module():
    return importlib.import_module("scripts.analysis.run_truth_veto_closure")


def _rotation_z(angle: float) -> list[list[float]]:
    import math

    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _surface(volume: int, layer: int, sensor: int, tx: float, ty: float, rz: float):
    return {
        "ID": f"vol={volume}|lay={layer}|sen={sensor}",
        "Translation": [tx, ty, 0.0],
        "RotationMatrix": _rotation_z(rz),
    }


def _write_geometry(path: Path, scale: float, *, reverse: bool = False) -> None:
    rows = []
    for volume, layer in TARGET_LAYERS:
        for sensor in range(1, 7):
            sign = -1.0 if sensor % 2 else 1.0
            rows.append(
                _surface(
                    volume,
                    layer,
                    sensor,
                    scale * sign * (0.2 + sensor * 0.01),
                    scale * sign * (0.3 + sensor * 0.01),
                    scale * sign * (0.01 + sensor * 0.001),
                )
            )
    if reverse:
        rows.reverse()
    path.write_text(json.dumps(rows), encoding="utf-8")


def _geometries(tmp_path: Path, candidate_scale: float):
    nominal = tmp_path / "nominal.json"
    misaligned = tmp_path / "misaligned.json"
    candidate = tmp_path / "candidate.json"
    _write_geometry(nominal, 0.0)
    _write_geometry(misaligned, 1.0)
    _write_geometry(candidate, candidate_scale, reverse=True)
    return nominal, misaligned, candidate


def _compute(module, tmp_path: Path, candidate_scale: float):
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale)
    return module.compute_geometry_truth_veto(
        misaligned_geometry_path=misaligned,
        nominal_geometry_path=nominal,
        candidate_geometry_path=candidate,
        target_layers=TARGET_LAYERS,
        active_dof=ACTIVE_DOF,
        bootstrap_seed=6006,
        n_bootstrap=200,
    )


def test_geometry_candidate_improves_all_six_layer_dof_metrics(tmp_path):
    result = _compute(_module(), tmp_path, candidate_scale=0.25)

    assert set(result["metrics"]) == {
        "17_8_tx", "17_8_ty", "17_8_rz",
        "24_4_tx", "24_4_ty", "24_4_rz",
    }
    assert result["verdict"]["passed"] is True
    assert result["verdict"]["failed_metrics"] == []
    for metric in result["metrics"].values():
        assert metric["n_blocks"] == 6
        assert metric["nominal_ratio"] == pytest.approx(0.25)
        assert metric["ci_upper"] < 1.0


def test_one_worse_layer_dof_fails_the_candidate(tmp_path):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=0.25)
    rows = json.loads(candidate.read_text(encoding="utf-8"))
    for row in rows:
        if row["ID"].startswith("vol=24|lay=4"):
            row["Translation"][1] *= 8.0
    candidate.write_text(json.dumps(rows), encoding="utf-8")

    result = module.compute_geometry_truth_veto(
        misaligned_geometry_path=misaligned,
        nominal_geometry_path=nominal,
        candidate_geometry_path=candidate,
        target_layers=TARGET_LAYERS,
        active_dof=ACTIVE_DOF,
        bootstrap_seed=6006,
        n_bootstrap=200,
    )

    assert result["verdict"]["passed"] is False
    assert result["verdict"]["failed_metrics"] == ["24_4_ty"]


def test_ratio_equal_to_one_fails_strict_improvement(tmp_path):
    result = _compute(_module(), tmp_path, candidate_scale=1.0)

    assert result["verdict"]["passed"] is False
    assert set(result["verdict"]["failed_metrics"]) == set(result["metrics"])


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonfinite"])
def test_invalid_candidate_geometry_fails_closed(tmp_path, corruption):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=0.25)
    rows = json.loads(candidate.read_text(encoding="utf-8"))
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["Translation"][0] = float("nan")
    candidate.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError):
        module.compute_geometry_truth_veto(
            misaligned_geometry_path=misaligned,
            nominal_geometry_path=nominal,
            candidate_geometry_path=candidate,
            target_layers=TARGET_LAYERS,
            active_dof=ACTIVE_DOF,
            bootstrap_seed=6006,
            n_bootstrap=20,
        )


def test_existing_run_directory_is_not_overwritten(tmp_path):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=0.25)
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.main(
            [
                "--misaligned-geometry", str(misaligned),
                "--nominal-geometry", str(nominal),
                "--candidate-geometry", str(candidate),
                "--run-dir", str(run_dir),
                "--output", str(run_dir / "truth_veto_result.json"),
            ]
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_scientific_failure_finalizes_completed_manifest(tmp_path):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=1.0)
    run_dir = tmp_path / "scientific-failure"
    output = run_dir / "truth_veto_result.json"

    status = module.main(
        [
            "--misaligned-geometry", str(misaligned),
            "--nominal-geometry", str(nominal),
            "--candidate-geometry", str(candidate),
            "--run-dir", str(run_dir),
            "--output", str(output),
            "--n-bootstrap", "20",
        ]
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert status == 1
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["verdict"]["passed"] is False
    assert output.is_file()


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_exception_and_interrupt_finalize_failed_manifest(tmp_path, monkeypatch, failure):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=0.25)
    run_dir = tmp_path / type(failure).__name__

    def fail(**_kwargs):
        raise failure

    monkeypatch.setattr(module, "compute_geometry_truth_veto", fail)
    with pytest.raises(type(failure)):
        module.main(
            [
                "--misaligned-geometry", str(misaligned),
                "--nominal-geometry", str(nominal),
                "--candidate-geometry", str(candidate),
                "--run-dir", str(run_dir),
                "--output", str(run_dir / "truth_veto_result.json"),
                "--n-bootstrap", "20",
            ]
        )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["error_type"] == type(failure).__name__


def test_capture_source_snapshot_flag_is_forwarded(tmp_path, monkeypatch):
    module = _module()
    nominal, misaligned, candidate = _geometries(tmp_path, candidate_scale=0.25)
    captured = {}

    def fake_start(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after manifest arguments")

    monkeypatch.setattr(module, "write_experiment_manifest", fake_start)
    with pytest.raises(RuntimeError):
        module.main(
            [
                "--misaligned-geometry", str(misaligned),
                "--nominal-geometry", str(nominal),
                "--candidate-geometry", str(candidate),
                "--run-dir", str(tmp_path / "snapshot"),
                "--output", str(tmp_path / "snapshot" / "truth_veto_result.json"),
                "--capture-source-snapshot",
            ]
        )
    assert captured["capture_source_snapshot"] is True
