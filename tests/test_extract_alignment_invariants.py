from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "extract_alignment_invariants.py"
LAYERS = "16:6,23:4"


def run(log_path, mode=("surface"), layers=LAYERS):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(log_path), "--mode", mode, "--layers", layers],
        capture_output=True,
        text=True,
    )


def _epoch_line(epoch, *, weights):
    base = (
        f"Epoch [{epoch:03d}/050] | LR: 0.010000 | "
        f"Chi2: 1.2345 | L2: 0.000012 | Total: 1.2346"
    )
    layers = " | v16l6=1.11 | v23l4=2.22"
    if not weights:
        return base + layers
    weight_fields = " | W: mean=1.000 std=0.000 min=1.000"
    weight_layers = " | Wv16l6: raw=1.000 norm=1.000 | Wv23l4: raw=1.000 norm=1.000"
    return base + weight_fields + layers + weight_layers


def _truth_section():
    lines = ["", "=== ML vs TRUE misalignment (local frame, per sensor) ==="]
    for volume, layer in ((16, 6), (23, 4)):
        lines.append(f"  vol={volume} lay={layer} (n=93)")
        for name in ("tx", "ty", "tz", "rx", "ry", "rz"):
            lines.append(
                f"    {name}: corr=+0.900  recovery=+0.800  "
                f"|true|rms=0.5000mm  |pred|rms=0.4000mm"
            )
    return lines


def _tables(*, new_labels):
    label = (
        "(n0=1234, n1=567)" if new_labels else "(n=1234)"
    )
    lines = ["", "=== Per-layer residual mean (training data) ==="]
    lines.append("  Layer                   X_before     X_after    Y_before     Y_after")
    for volume, layer, x_after in ((16, 6, "0.0012"), (23, 4, "0.0034")):
        before = "0.5000" if not new_labels else "0.4000"  # allowed to change
        lines.append(
            f"  vol={volume} lay={layer} {label}     {before}mm     {x_after}mm"
            f"     0.6000mm     0.0050mm"
        )
    lines.append("")
    lines.append("=== Per-layer residual RMS (training data) ===")
    lines.append("  Layer                   X_before     X_after    Y_before     Y_after")
    for volume, layer in ((16, 6), (23, 4)):
        # deliberately different from the mean table: std -> true RMS is expected
        lines.append(
            f"  vol={volume} lay={layer} {label}     0.9999mm     0.8888mm"
            f"     0.7777mm     0.6666mm"
        )
    return lines


def write_surface_log(tmp_path, name, *, weights, new_labels):
    lines = [_epoch_line(e, weights=weights) for e in (1, 10, 20, 30, 40, 50)]
    lines += _truth_section()
    lines += _tables(new_labels=new_labels)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_free_log(tmp_path, name):
    lines = [
        "Joint [0000/100] chi2/component=5.4321 invalid=0.12%",
        "Joint [0099/100] chi2/component=1.2345 invalid=0.10%",
    ]
    lines += _truth_section()
    lines += _tables(new_labels=True)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_old_and_new_log_formats_extract_identical_invariants(tmp_path):
    """The whole point of the extractor: the refactor must be invisible here."""
    old = write_surface_log(tmp_path, "old.log", weights=True, new_labels=False)
    new = write_surface_log(tmp_path, "new.log", weights=False, new_labels=True)

    old_result = run(old)
    new_result = run(new)

    assert old_result.returncode == 0, old_result.stderr
    assert new_result.returncode == 0, new_result.stderr
    assert old_result.stdout == new_result.stdout


def test_weight_diagnostic_tokens_are_not_mistaken_for_layer_chi2(tmp_path):
    """``Wv16l6: raw=`` must not be captured as a ``v16l6=`` chi2 token."""
    old = write_surface_log(tmp_path, "old.log", weights=True, new_labels=False)

    result = run(old)

    assert result.returncode == 0, result.stderr
    iteration_lines = [l for l in result.stdout.splitlines() if l.startswith("ITER")]
    assert iteration_lines
    for line in iteration_lines:
        assert line.count("v16l6=") == 1
        assert line.count("v23l4=") == 1
        assert "Wv" not in line


def test_extracts_expected_record_kinds(tmp_path):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)

    result = run(log)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert sum(l.startswith("ITER ") for l in lines) == 6
    assert sum(l.startswith("TRUTH ") for l in lines) == 12  # 2 layers x 6 DOF
    assert "MEAN_X_AFTER vol=16 lay=6 0.0012" in lines
    assert "MEAN_X_AFTER vol=23 lay=4 0.0034" in lines


def test_free_mode_accepts_joint_iteration_records(tmp_path):
    log = write_free_log(tmp_path, "free.log")

    result = run(log, mode="free")

    assert result.returncode == 0, result.stderr
    assert "ITER 0000 chi2=5.4321 invalid=0.12" in result.stdout
    assert "ITER 0099 chi2=1.2345 invalid=0.10" in result.stdout


def test_surface_mode_rejects_free_log(tmp_path):
    """Guards against silently extracting nothing when the mode is wrong."""
    log = write_free_log(tmp_path, "free.log")

    result = run(log, mode="surface")

    assert result.returncode != 0
    assert "no iteration records" in result.stderr


@pytest.mark.parametrize(
    "drop, expected",
    [
        ("=== Per-layer residual mean", "Per-layer residual mean"),
        ("=== Per-layer residual RMS", "Per-layer residual RMS"),
        ("=== ML vs TRUE misalignment", "ML vs TRUE misalignment"),
    ],
)
def test_missing_section_header_fails_closed(tmp_path, drop, expected):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)
    kept = [l for l in log.read_text(encoding="utf-8").splitlines() if drop not in l]
    log.write_text("\n".join(kept) + "\n", encoding="utf-8")

    result = run(log)

    assert result.returncode != 0
    assert expected in result.stderr


def test_incomplete_epoch_record_set_fails_closed(tmp_path):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)
    kept = [
        l for l in log.read_text(encoding="utf-8").splitlines()
        if not l.startswith("Epoch [050")
    ]
    log.write_text("\n".join(kept) + "\n", encoding="utf-8")

    result = run(log)

    assert result.returncode != 0
    assert "do not match the expected" in result.stderr


def test_missing_target_layer_row_fails_closed(tmp_path):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)

    result = run(log, layers="16:6,23:4,28:2")

    assert result.returncode != 0
    assert "28:2" in result.stderr


def test_missing_truth_dof_fails_closed(tmp_path):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)
    kept = [
        l for l in log.read_text(encoding="utf-8").splitlines()
        if not l.strip().startswith("rz: corr=")
    ]
    log.write_text("\n".join(kept) + "\n", encoding="utf-8")

    result = run(log)

    assert result.returncode != 0
    assert "missing DOF 'rz'" in result.stderr


def test_epoch_line_missing_chi2_fails_closed(tmp_path):
    log = write_surface_log(tmp_path, "run.log", weights=False, new_labels=True)
    text = log.read_text(encoding="utf-8").replace("Chi2: 1.2345 | ", "", 1)
    log.write_text(text, encoding="utf-8")

    result = run(log)

    assert result.returncode != 0
    assert "missing 'Chi2'" in result.stderr
