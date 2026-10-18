import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("compare_pt100_alignment", "scripts/analysis/compare_pt100_alignment.py")
schur_driver = load("run_pt100_schur_calibration", "scripts/analysis/run_pt100_schur_calibration.py")


def summary(aligned_rms, injected_rms):
    dof = {name: {"aligned_difference": {"rms": aligned_rms[name]},
                  "misaligned_difference": {"rms": injected_rms[name]}}
           for name in ("tx", "ty", "rz")}
    return {"by_layer": {"vol=17|lay=8": {"dof": dof}}}


def test_geometry_rows_convert_to_micron_and_mrad():
    injected = {"tx": 0.5, "ty": 0.52, "rz": 0.02}
    rows = compare.geometry_rows(
        {"schur": summary({"tx": 0.003, "ty": 0.004, "rz": 1.4e-4}, injected),
         "acts": summary({"tx": 0.0025, "ty": 0.0034, "rz": 1.2e-4}, injected)},
        [(17, 8)])
    assert [r["dof"] for r in rows] == ["tx", "ty", "rz"]
    assert rows[1]["schur"] == pytest.approx(4.0)
    assert rows[1]["acts"] == pytest.approx(3.4)
    assert rows[2]["unit"] == "mrad" and rows[2]["acts"] == pytest.approx(0.12)
    assert rows[0]["injected"] == pytest.approx(500.0)


def test_geometry_rows_reject_different_injected_draws():
    with pytest.raises(ValueError, match="injected"):
        compare.geometry_rows(
            {"schur": summary({"tx": 0, "ty": 0, "rz": 0}, {"tx": 0.5, "ty": 0.5, "rz": 0.02}),
             "acts": summary({"tx": 0, "ty": 0, "rz": 0}, {"tx": 0.4, "ty": 0.5, "rz": 0.02})},
            [(17, 8)])


def test_distribution_summary_rms_includes_mean():
    row = compare.distribution_summary([1.0, 3.0])
    assert row["count"] == 2 and row["mean"] == 2.0 and row["std"] == 1.0
    assert row["rms"] == pytest.approx(math.sqrt(5.0))
    with pytest.raises(ValueError):
        compare.distribution_summary([])
    with pytest.raises(ValueError):
        compare.distribution_summary([1.0, float("nan")])


ACTS_LOG = """\
Evaluator        WARN  +++ noise
23:59:50    Sequencer      INFO      Starting event loop with 1 threads
23:59:58    AlignmentAlg   INFO      Finalizing alignment with 64 collected tracks from all events
00:10:00    Alignment      INFO      iIter = 0, total chi2 = 10, total measurementDim = 8 and average chi2/ndf = 1.5
00:20:02    Alignment      INFO      iIter = 1, total chi2 = 9, total measurementDim = 8 and average chi2/ndf = 1.25
00:20:02    Alignment      INFO      Alignment has converged with change of chi2/ndf < 0.0001 in the last 5 iterations after 1 iteration(s)
00:20:05    AlignmentAlg   INFO      Alignment finished successfully!
"""


def test_parse_acts_alignment_log_unwraps_midnight():
    parsed = compare.parse_acts_alignment_log(ACTS_LOG)
    assert parsed["event_loop_seconds"] == 8
    assert parsed["finalize_seconds"] == 20 * 60 + 7
    assert parsed["iterations"] == 2
    assert parsed["iteration_end_seconds"] == [602, 1204]
    assert parsed["average_chi2_ndf"] == [1.5, 1.25]
    assert parsed["converged"] and parsed["collected_tracks"] == 64


def test_parse_acts_alignment_log_without_finalize():
    parsed = compare.parse_acts_alignment_log("10:00:00    Sequencer      INFO      Starting event loop with 1 threads\n")
    assert parsed["event_loop_seconds"] is None and parsed["iterations"] == 0
    assert not parsed["converged"]


FULL_LOG = """\
STEP 1: SIMULATION (using NOMINAL geometry)
10:00:30    Sequencer      INFO      Processed 100 events in 30.5 s (wall clock)
STEP 2: ALIGNMENT (starting from MISALIGNED geometry)
""" + ACTS_LOG + """\
00:20:05    Sequencer      INFO      Processed 100 events in 1215.25 s (wall clock)
STEP 3: RECONSTRUCTION WITH NOMINAL GEOMETRY
00:21:00    Sequencer      INFO      Processed 100 events in 40.0 s (wall clock)
STEP 4: RECONSTRUCTION WITH MISALIGNED GEOMETRY
00:22:00    Sequencer      INFO      Processed 100 events in 52.5 s (wall clock)
STEP 5: RECONSTRUCTION WITH ALIGNED GEOMETRY
00:23:00    Sequencer      INFO      Processed 100 events in 51.0 s (wall clock)
"""


def test_full_workflow_records_split_steps():
    timings, acts_log = compare.full_workflow_records(FULL_LOG)
    assert timings["alignment"]["wall_seconds"] == 1215.25
    assert timings["misaligned"]["wall_seconds"] == 52.5
    assert timings["alignment"]["process_cpu_seconds"] is None
    assert acts_log["event_loop_seconds"] == 8 and acts_log["iterations"] == 2
    assert acts_log["converged"] and acts_log["collected_tracks"] == 64


def test_full_workflow_records_require_each_step_once():
    with pytest.raises(ValueError):
        compare.full_workflow_records(FULL_LOG.replace("STEP 4:", "STEP 9:"))
    with pytest.raises(ValueError):
        compare.full_workflow_records(FULL_LOG.replace("in 52.5 s", "in s"))


def test_full_workflow_reconstruction_is_resolved_from_its_manifest(tmp_path):
    manifest = {"status": "completed", "metrics": {"workflow_config": {}},
                "outputs": {"reconstruction_misaligned": str(tmp_path / "mis")}}
    assert compare.is_full_workflow(manifest)
    assert not compare.is_full_workflow({"status": "completed", "metrics": {}})
    assert compare.full_workflow_output(manifest, "reconstruction_misaligned") == tmp_path / "mis"
    with pytest.raises(ValueError):
        compare.full_workflow_output(manifest, "reconstruction_aligned")


def test_markdown_table_shape():
    text = compare.markdown_table(["a", "b"], [[1, 2]])
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_stage_timer_accumulates_and_captures_result():
    wall = iter([0.0, 2.0, 20.0, 25.0])
    cpu = iter([10.0, 11.0, 5.0, 6.0])
    timer = schur_driver.StageTimer(lambda: next(wall), lambda: next(cpu))
    sink = {}
    wrapped = timer.wrap("seed", lambda x: x * 2, sink)
    assert wrapped(3) == 6 and wrapped(4) == 8
    assert sink["seed"] == 8
    assert timer.stages["seed"] == {"wall_seconds": 7.0, "cpu_seconds": 2.0, "calls": 2}


def test_stage_timer_records_failed_call():
    wall, cpu = iter([0.0, 4.0]), iter([1.0, 2.0])
    timer = schur_driver.StageTimer(lambda: next(wall), lambda: next(cpu))

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        timer.wrap("refit", boom)()
    assert timer.stages["refit"]["wall_seconds"] == 4.0


def test_orchestrator_args_select_schur_lm_with_validated_seed_iterations():
    args = schur_driver.ORCHESTRATOR_ARGS
    assert args[args.index("--joint-solver") + 1] == "schur-lm"
    assert args[args.index("--joint-smt-seed-iters") + 1] == "2000"


def test_layout_shares_one_misalignment_and_separates_arms():
    import pt100_alignment_layout as layout

    paths = layout.pt100_alignment_paths("/d", "Vol17_Lay8__Vol24_Lay4", "r")
    assert paths["misalignment"].parent == paths["acts_geometry"].parent == paths["alignment"]
    assert paths["schur_geometry"].parent == paths["schur_single_thread"]
    assert paths["schur_single_thread"] != paths["schur_multi_thread"]
    assert paths["simulation"] == Path("/d/simulation/material-helix-20261014-r2/g3_pt100_material_on")
    assert paths["nominal"].parent.parent == paths["misaligned"].parent.parent == paths["reconstruction"]


def stage_dirs(tmp_path, alignment_status="completed", misaligned=None, outputs=None):
    import json

    paths = {"alignment": tmp_path / "alignment", "misaligned": tmp_path / "misaligned"}
    for path in paths.values():
        path.mkdir()
    (paths["alignment"] / "manifest.json").write_text(
        json.dumps({"status": alignment_status, "outputs": outputs or {}}))
    if misaligned is not None:
        (paths["misaligned"] / "manifest.json").write_text(json.dumps({"status": misaligned}))
    return paths


def test_inputs_accept_per_stage_misaligned_manifest(tmp_path):
    schur_driver.require_completed_inputs(stage_dirs(tmp_path, misaligned="completed"))
    running = tmp_path / "running"
    running.mkdir()
    with pytest.raises(ValueError):
        schur_driver.require_completed_inputs(stage_dirs(running, misaligned="running"))


def test_inputs_accept_full_workflow_manifest_covering_misaligned(tmp_path):
    misaligned = str((tmp_path / "misaligned").resolve())
    schur_driver.require_completed_inputs(
        stage_dirs(tmp_path, outputs={"reconstruction_misaligned": misaligned}))


def test_inputs_reject_unattested_misaligned_reconstruction(tmp_path):
    with pytest.raises(ValueError):
        schur_driver.require_completed_inputs(
            stage_dirs(tmp_path, outputs={"reconstruction_misaligned": "/elsewhere"}))
    other = tmp_path / "running"
    other.mkdir()
    misaligned = str((other / "misaligned").resolve())
    with pytest.raises(ValueError):
        schur_driver.require_completed_inputs(
            stage_dirs(other, alignment_status="running",
                       outputs={"reconstruction_misaligned": misaligned}))


def geometry_record(dx=0.0, dr=0.0):
    return [{"ID": "vol=17|lay=8|sen=1", "Translation": [1.0 + dx, 2.0, 3.0],
             "RotationMatrix": [[1.0, dr, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}]


def test_thread_consistency_uses_schur_residual_scale():
    rows = [{"dof": "tx", "schur": 3.0}, {"dof": "ty", "schur": 4.0}, {"dof": "rz", "schur": 0.14}]
    small = compare.geometry_max_difference(geometry_record(), geometry_record(dx=1e-8, dr=1e-9))
    assert compare.thread_consistency(small, rows)["consistent"]
    large = compare.geometry_max_difference(geometry_record(), geometry_record(dx=1e-4))
    verdict = compare.thread_consistency(large, rows)
    assert not verdict["consistent"] and verdict["translation_limit_um"] == pytest.approx(0.03)


def test_geometry_max_difference_requires_same_surfaces():
    other = [{**geometry_record()[0], "ID": "vol=24|lay=4|sen=1"}]
    with pytest.raises(ValueError, match="different surfaces"):
        compare.geometry_max_difference(geometry_record(), other)


tail = load("pt100_sensor_tail", "scripts/analysis/pt100_sensor_tail.py")


def test_tail_rows_drop_low_hit_sensors_from_kept_rms():
    per_sensor, hits = {}, {}
    for sensor in range(10):
        hits[(17, 8, sensor)] = 1 if sensor == 0 else 20
        for dof in ("tx", "ty", "rz"):
            acts = 0.5 if sensor == 0 else 0.004
            per_sensor[(17, 8, sensor, dof)] = (acts, 0.004)
    rows = tail.tail_rows(per_sensor, hits)
    tx = rows[0]
    assert tx["dof"] == "tx" and tx["hits_cut"] == 1 and tx["sensors_kept"] == 9
    assert tx["acts_rms"] > 10 * tx["schur_rms"]
    assert tx["acts_rms_kept"] == pytest.approx(4.0) and tx["schur_rms_kept"] == pytest.approx(4.0)
    assert tx["schur_better_fraction"] == pytest.approx(0.1)
