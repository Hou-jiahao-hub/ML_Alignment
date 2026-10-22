import importlib
import json

import pytest

from mlalign.reporting import write_alignment_summary
from mlalign.timing import RunTimings


def test_stage_records_wall_and_all_thread_cpu_seconds(monkeypatch):
    module = importlib.import_module('mlalign.timing')
    wall = iter([10.0, 12.0])
    cpu = iter([20.0, 25.0])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(wall, 12.0))
    monkeypatch.setattr(module.time, 'process_time', lambda: next(cpu, 25.0))
    timings = RunTimings()
    assert timings.call('solve', lambda: 7) == 7
    row = timings.snapshot()['stages']['solve'][0]
    assert row == {'wall_seconds': 2.0, 'process_cpu_seconds': 5.0,
                   'status': 'completed', 'error': None}


@pytest.mark.parametrize('exception,status', [(ValueError('broken'), 'failed'),
                                               (KeyboardInterrupt(), 'interrupted')])
def test_failure_keeps_original_exception_and_partial_timing(exception, status):
    timings = RunTimings()
    with pytest.raises(type(exception)) as caught:
        with timings.stage('root_load'):
            raise exception
    assert caught.value is exception
    row = timings.snapshot()['stages']['root_load'][0]
    assert row['status'] == status
    assert row['wall_seconds'] >= 0


def test_pipeline_wrapper_and_summary_preserve_timings(monkeypatch, tmp_path):
    module = importlib.import_module('mlalign.pipeline')
    def fake(config, timings):
        return timings.call('solve', lambda: {'solver': 'schur-lm'})
    monkeypatch.setattr(module, '_run_alignment', fake)
    result = module.run_alignment(object())
    destination = tmp_path / 'summary.json'
    write_alignment_summary(result, destination)
    payload = json.loads(destination.read_text())
    assert set(payload['timings']['stages']) == {'solve', 'pipeline_total'}
    assert payload['timings']['stages']['pipeline_total'][0]['status'] == 'completed'


@pytest.mark.parametrize("exception,status", [
    (RuntimeError("input failed"), "failed"),
    (KeyboardInterrupt(), "interrupted"),
])
def test_cli_saves_failure_timings_only_in_summary(monkeypatch, tmp_path, exception, status):
    cli = importlib.import_module("mlalign.cli")
    def broken(config, *, timings):
        with timings.stage("root_load"):
            raise exception
    monkeypatch.setattr(cli, "run_alignment", broken)
    destination = tmp_path / "schur_lm_summary.json"
    with pytest.raises(type(exception)) as caught:
        cli.main(["--input-geom", "input.json", "--states", "states.root",
                  "--nominal-full-geom", "nominal.json", "--output-geom", "out.json",
                  "--output-summary", str(destination)])
    assert caught.value is exception
    payload = json.loads(destination.read_text())
    assert payload["execution_status"] == status
    assert payload["timings"]["stages"]["root_load"][0]["status"] == status
    assert "schur_lm_summary" not in payload
    assert list(tmp_path.iterdir()) == [destination]


def test_cli_saves_success_timings_only_in_summary(monkeypatch, tmp_path):
    cli = importlib.import_module("mlalign.cli")
    pipeline = importlib.import_module("mlalign.pipeline")
    monkeypatch.setattr(pipeline, "_run_alignment", lambda config, timings: {"solver": "schur-lm"})
    monkeypatch.setattr(cli, "run_alignment", pipeline.run_alignment)
    destination = tmp_path / "schur_lm_summary.json"
    assert cli.main(["--input-geom", "input.json", "--states", "states.root",
                     "--nominal-full-geom", "nominal.json", "--output-geom", "out.json",
                     "--output-summary", str(destination)]) == 0
    payload = json.loads(destination.read_text())
    assert payload["execution_status"] == "completed"
    assert payload["execution_error"] is None
    assert payload["timings"]["stages"]["pipeline_total"][0]["status"] == "completed"
    assert list(tmp_path.iterdir()) == [destination]


def test_reporting_retains_timing_and_execution_metadata(tmp_path):
    destination = tmp_path / "summary.json"
    timing = {
        "schema_version": 1,
        "unit": "seconds",
        "stages": {"pipeline_total": [{
            "wall_seconds": 2.0,
            "process_cpu_seconds": 5.0,
            "status": "completed",
            "error": None,
        }]},
    }
    write_alignment_summary({
        "timings": timing,
        "execution_status": "completed",
        "execution_error": None,
    }, destination)
    payload = json.loads(destination.read_text())
    assert payload.get("timings") == timing
    assert payload.get("execution_status") == "completed"
    assert "execution_error" in payload


def test_cli_missing_input_records_total_failure_in_summary(tmp_path):
    cli = importlib.import_module("mlalign.cli")
    destination = tmp_path / "schur_lm_summary.json"
    missing_geometry = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Missing input geometry"):
        cli.main([
            "--input-geom", str(missing_geometry),
            "--states", str(tmp_path / "missing.root"),
            "--nominal-full-geom", str(tmp_path / "nominal.json"),
            "--output-geom", str(tmp_path / "aligned.json"),
            "--output-summary", str(destination),
        ])

    payload = json.loads(destination.read_text())
    assert payload["execution_status"] == "failed"
    assert "FileNotFoundError" in payload["execution_error"]
    assert payload["timings"]["stages"]["pipeline_total"][0]["status"] == "failed"
    assert list(tmp_path.iterdir()) == [destination]
