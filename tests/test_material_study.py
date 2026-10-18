"""Exercise the ACTS adapter's study orchestration without loading ACTS."""
import ast
import json
import time
import os
import hashlib
import __future__
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest

from scripts.experiment_manifest import write_experiment_manifest, finalize_experiment_manifest
from scripts.project_config import load_project_config


def adapter(extra=()):
    source = Path(__file__).resolve().parents[1] / "scripts/alignment_odd.py"
    tree = ast.parse(source.read_text())
    names = {"material_study_paths", "run_material_study_stage", *extra}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), "Missing material study orchestration"
    namespace = dict(Path=Path, json=json, time=time, os=os, hashlib=hashlib,
                     traceback=traceback,
                     redirect_stdout=redirect_stdout, redirect_stderr=redirect_stderr,
                     sys=SimpleNamespace(executable="python", argv=[], stdout=__import__('sys').stdout,
                                         stderr=__import__('sys').stderr),
                     PROJECT_CONFIG=load_project_config(),
                     write_experiment_manifest=write_experiment_manifest,
                     finalize_experiment_manifest=finalize_experiment_manifest)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace)


def test_study_paths_keep_groups_below_target(tmp_path):
    m = adapter()
    sim, reco = m.material_study_paths(tmp_path, "17_24", "run1", "g1")
    assert sim == tmp_path / "simulation/run1/g1"
    assert reco == tmp_path / "reconstruction/nominal/17_24/run1/g1"


def test_failed_stage_preserves_elapsed_time_and_manifest(tmp_path):
    m = adapter()
    def prepare():
        time.sleep(.002)
        raise RuntimeError("detector unavailable")
    with pytest.raises(RuntimeError, match="detector unavailable"):
        m.run_material_study_stage(tmp_path / "failed", "simulation", prepare,
                                   inputs={}, config={"pt_gev": 10})
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    timing = json.loads((tmp_path / "failed/timings.json").read_text())
    assert manifest["status"] == "failed"
    assert timing["setup_seconds"] > 0
    assert timing["run_seconds"] == 0
    assert timing["wall_seconds"] == timing["setup_seconds"] + timing["run_seconds"]


def test_stage_writes_artifact_and_refuses_reuse(tmp_path):
    m = adapter()
    out = tmp_path / "simulation"
    class Sequencer:
        def run(self):
            (out / "particles.root").write_bytes(b"actual sequencer output")
    m.run_material_study_stage(out, "simulation", Sequencer, inputs={}, config={})
    assert (out / "particles.root").read_bytes() == b"actual sequencer output"
    assert json.loads((out / "manifest.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        m.run_material_study_stage(out, "simulation", Sequencer, inputs={}, config={})


def test_simulation_passes_fixed_transverse_momentum_and_material_to_pipeline(tmp_path, monkeypatch):
    import sys
    from collections import namedtuple
    m = adapter(("runSimulation",))
    function = m.runSimulation
    fake = ModuleType("acts.examples.simulation")
    fake.MomentumConfig = namedtuple("MomentumConfig", "min max transverse")
    for name in ("ParticleConfig", "EtaConfig", "PhiConfig"):
        setattr(fake, name, lambda *args, **kwargs: (args, kwargs))
    def gun(seq, *configs, **kwargs):
        momentum = configs[2]
        seq.output.update(min=momentum.min, max=momentum.max, transverse=momentum.transverse)
    def fatras(seq, *args, **kwargs):
        seq.output["interactions"] = kwargs["enableInteractions"]
    fake.addParticleGun, fake.addFatras = gun, fatras
    fake.addDigitization = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "acts.examples.simulation", fake)
    function.__globals__.update(u=SimpleNamespace(GeV=1., degree=1., mm=1., ns=1.),
        acts=SimpleNamespace(PdgParticle=SimpleNamespace(eMuon=13), Vector4=lambda *a: a,
            examples=SimpleNamespace(RandomNumbers=lambda **k: None, GaussianVertexGenerator=lambda **k: None),
            logging=SimpleNamespace(getLogger=lambda *a: SimpleNamespace(info=lambda *a: None))))
    sequence = SimpleNamespace(output={})
    assert function(None, None, None, tmp_path, s=sequence, fixedPtGeV=10., enableInteractions=False) is sequence
    assert sequence.output == {"min": 10., "max": 10., "transverse": True, "interactions": False}


def test_failed_writer_is_closed_before_manifest_hashing(tmp_path):
    import gc
    m, output = adapter(), tmp_path / "failed_writer"
    class Sequencer:
        def run(self):
            (output / "state.root").write_bytes(b"not closed")
            raise RuntimeError("writer failed")
        def __del__(self):
            (output / "state.root").write_bytes(b"closed")
    with pytest.raises(RuntimeError, match="writer failed"):
        m.run_material_study_stage(output, "reconstruction", Sequencer, inputs={}, config={})
    gc.collect()
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / "state.root").read_bytes() == b"closed"
    assert manifest["metrics"]["output_sha256"]["state.root"] == hashlib.sha256(b"closed").hexdigest()


def test_partially_prepared_writer_is_closed_before_manifest_hashing(tmp_path):
    import gc
    m, output = adapter(), tmp_path / "failed_setup_writer"
    class Writer:
        def __init__(self):
            (output / "state.root").write_bytes(b"not closed")
        def __del__(self):
            (output / "state.root").write_bytes(b"closed")
    def prepare():
        writer = Writer()
        raise RuntimeError("digitization setup failed")
    with pytest.raises(RuntimeError, match="digitization setup failed"):
        m.run_material_study_stage(output, "simulation", prepare, inputs={}, config={})
    gc.collect()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metrics"]["output_sha256"]["state.root"] == hashlib.sha256(b"closed").hexdigest()


def test_existing_subcommands_execute_the_pinned_nominal_arms(tmp_path):
    m = adapter(("run_material_study",))
    source = Path(__file__).resolve().parents[1] / "scripts/alignment_odd.py"
    tree = ast.parse(source.read_text())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id.startswith("MATERIAL_STUDY_")}
    namespace = m.run_material_study.__globals__
    config = SimpleNamespace(data_root=tmp_path, odd_digi_config=tmp_path / "digi.json")
    config.odd_digi_config.write_text('{}')
    geometry = tmp_path / "nominal.json"
    geometry.write_text('[]')
    namespace.update(constants)
    namespace.update(PROJECT_CONFIG=config, TARGET_LAYER_DIR="from_config", u=SimpleNamespace(T=1),
        acts=SimpleNamespace(ConstantBField=lambda value: value, Vector3=lambda *args: args),
        data_path=lambda value: geometry, run_material_study_stage=lambda out, stage, prepare, **kw: prepare().run())
    def detector(*, misaligned):
        assert not misaligned
        return SimpleNamespace(trackingGeometry=lambda: "nominal geometry")
    namespace["get_configured_open_data_detector"] = detector
    def simulate(**kwargs):
        output = kwargs["outputDir"]
        output.mkdir(parents=True)
        def run():
            (output / "manifest.json").write_text(json.dumps({"status": "completed"}))
            (output / "parameters.json").write_text(json.dumps({
                "pt": kwargs["fixedPtGeV"], "material": kwargs["enableInteractions"],
                "events": kwargs["numEvents"], "geometry": kwargs["trackingGeometry"]}))
        return SimpleNamespace(run=run)
    def reconstruct(**kwargs):
        assert kwargs["finalOutputDir"] and not kwargs["overwriteOutputs"]
        output = kwargs["outputDir"]
        output.mkdir(parents=True)
        return SimpleNamespace(run=lambda: (output / "mode.txt").write_text(kwargs["mode"]))
    namespace.update(runSimulation=simulate, runReconstruction=reconstruct)
    m.run_material_study("simulation")
    m.run_material_study("reconstruction")
    groups = ["g1_pt10_material_on", "g2_pt10_material_off", "g3_pt100_material_on"]
    for group, pt, material in zip(groups, [10., 10., 100.], [True, False, True]):
        sim, reco = m.material_study_paths(tmp_path, "from_config", "material-helix-20261014-r2", group)
        assert json.loads((sim / "parameters.json").read_text()) == {
            "pt": pt, "material": material, "events": 20000, "geometry": "nominal geometry"}
        assert (reco / "mode.txt").read_text() == "nominal"
