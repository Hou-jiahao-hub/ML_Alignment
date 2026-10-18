from pathlib import Path
from typing import Optional
import json
import random
import numpy as np
import ctypes
import os
import argparse
import sys
import time
import hashlib
import traceback
from contextlib import redirect_stdout, redirect_stderr

import acts
import acts.examples
import acts.examples.alignment
from acts.examples.alignment import AlignmentDecorator

from alignment_artifact_layout import (
    PROJECT_CONFIG,
    TARGET_LAYER_DIR,
    TARGET_LAYER_SUFFIX,
    artifact_filename,
    data_path,
    target_subdir,
    target_volume_map,
)
from experiment_manifest import (
    finalize_experiment_manifest,
    write_experiment_manifest,
)
# Pinned pT=100 GeV ACTS-Alignment vs Schur-LM study (ACTS-free shared layout).
import pt100_alignment_layout as pt100
from pt100_alignment_layout import PT100_STAGES
from project_config import (
    absolute_project_path,
    add_config_argument,
    load_project_config,
    resolve_project_path,
)

u = acts.UnitConstants

# Deliberately hardcoded: simulation/reconstruction gain no CLI switches.
MATERIAL_STUDY_RUN = "material-helix-20261014-r2"
MATERIAL_STUDY_EVENTS = 20_000
MATERIAL_STUDY_CASES = (
    ("g1_pt10_material_on", 10.0, True),
    ("g2_pt10_material_off", 10.0, False),
    ("g3_pt100_material_on", 100.0, True),
)


def material_study_paths(data_root, target_dir, run_id, group):
    root = Path(data_root)
    return (root / "simulation" / run_id / group,
            root / "reconstruction" / "nominal" / target_dir / run_id / group)


def run_material_study_stage(output, stage, prepare, *, inputs, config):
    """Own fresh outputs, provenance, native/Python logs and failure-safe timing."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to reuse study output: {output}")
    output.mkdir(parents=True)
    timings = {"schema_version": 1, "stage": stage, "setup_seconds": 0.0,
               "run_seconds": 0.0, "wall_seconds": 0.0,
               "excludes": "environment loading, manifest hashing and source archival"}
    timing_path = output / "timings.json"
    cpu_started = cpu_stopped = None
    manifest = write_experiment_manifest(
        output, run_id=f"{output.parent.name}/{output.name}/{stage}",
        command=[sys.executable, *sys.argv], config=PROJECT_CONFIG,
        inputs=inputs, outputs={"timings": timing_path, "log": output / f"{stage}.log"},
        metrics={"study_config": config}, capture_source_snapshot=True, allow_missing_inputs=True)
    status, error, phase, started = "failed", None, "setup_seconds", None
    try:
        with (output / f"{stage}.log").open("w") as log:
            sys.stdout.flush()
            sys.stderr.flush()
            saved = (os.dup(1), os.dup(2))
            try:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        started = time.perf_counter()
                        cpu_started = time.process_time()
                        sequencer = prepare()
                        timings[phase] = time.perf_counter() - started
                        phase, started = "run_seconds", time.perf_counter()
                        try:
                            sequencer.run()
                        finally:
                            del sequencer
                        timings[phase] = time.perf_counter() - started
                        cpu_stopped = time.process_time()
                        started = None
                    except BaseException as exc:
                        # Setup can also fail after creating writer-owning `s`.
                        # Release traceback frames while the native log is open.
                        traceback.clear_frames(exc.__traceback__)
                        raise
            finally:
                log.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                os.close(saved[0])
                os.close(saved[1])
        status = "completed"
    except BaseException as exc:
        if started is not None:
            timings[phase] = time.perf_counter() - started
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings["wall_seconds"] = timings["setup_seconds"] + timings["run_seconds"]
        # Process CPU over setup+run, summed over all threads: a thread-count-free check.
        if cpu_started is not None:
            timings["process_cpu_seconds"] = (cpu_stopped or time.process_time()) - cpu_started
        timing_path.write_text(json.dumps(timings, indent=2, allow_nan=False) + "\n")
        outputs = {p.name: p for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
        hashes = {}
        for name, path in outputs.items():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
        finalize_experiment_manifest(manifest, status=status, outputs=outputs,
                                     metrics={"timings": timings, "error": error,
                                              "output_sha256": hashes})


def run_material_study(stage):
    """Run the three pinned arms through an existing adapter subcommand."""
    config = {"run_id": MATERIAL_STUDY_RUN, "num_events": MATERIAL_STUDY_EVENTS,
              "random_seed": 42, "geometry": "nominal", "field_tesla": 2.0,
              "particle": "muon, randomized charge", "eta": [-3., 3.],
              "reconstruction": "unchanged TruthSmeared seed + Kalman"}
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    for group, pt, interactions in MATERIAL_STUDY_CASES:
        simulation, reconstruction = material_study_paths(
            PROJECT_CONFIG.data_root, TARGET_LAYER_DIR, MATERIAL_STUDY_RUN, group)
        output = simulation if stage == "simulation" else reconstruction
        inputs = {"digi_config": PROJECT_CONFIG.odd_digi_config,
                  "nominal_geometry": data_path("odd_transforms.json")}
        if stage == "reconstruction":
            if not (simulation / "manifest.json").exists():
                raise FileNotFoundError(f"missing completed simulation: {simulation}")
            if json.loads((simulation / "manifest.json").read_text())["status"] != "completed":
                raise ValueError(f"simulation is not completed: {simulation}")
            inputs["simulation_dir"] = simulation
        detectors = []
        def prepare():
            detector = get_configured_open_data_detector(misaligned=False)
            detectors.append(detector)  # Keep detector elements alive through Sequencer.run().
            common = dict(trackingGeometry=detector.trackingGeometry(), field=field,
                          digiConfigFile=PROJECT_CONFIG.odd_digi_config, outputDir=output,
                          detector=detector, numEvents=MATERIAL_STUDY_EVENTS)
            if stage == "simulation":
                return runSimulation(**common, fixedPtGeV=pt, enableInteractions=interactions)
            return runReconstruction(**common, inputDir=simulation, mode="nominal",
                                     finalOutputDir=True, overwriteOutputs=False)
        print(f"Starting {stage}: {group} -> {output}", flush=True)
        run_material_study_stage(output, stage, prepare, inputs=inputs,
                                 config={**config, "group": group, "pt_gev": pt,
                                         "fatras_interactions": interactions})
        print(f"Completed {stage}: {group}", flush=True)


class _DeferredRun:
    """Adapter for run_material_study_stage when the callee runs its own Sequencer."""

    def __init__(self, call):
        self._call = call

    def run(self):
        self._call()


def _require_completed(directory):
    manifest = Path(directory) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing completed stage: {directory}")
    if json.loads(manifest.read_text())["status"] != "completed":
        raise ValueError(f"stage is not completed: {directory}")


def link_pt100_nominal(paths):
    """Expose the existing g3 nominal reconstruction in the plot-script layout."""
    if paths["nominal"].exists():
        return
    _require_completed(paths["nominal_source"])
    paths["nominal"].mkdir(parents=True)
    for stem in ("trackstates_kf", "tracksummary_kf"):
        name = artifact_filename(stem, ".root")
        (paths["nominal"] / name).symlink_to(paths["nominal_source"] / name)


def full_workflow_manifest(alignment_dir):
    """The `full` workflow's manifest in `alignment_dir`, or None for any other run."""
    manifest = Path(alignment_dir) / "manifest.json"
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    return record if "workflow_config" in record.get("metrics", {}) else None


def run_pt100_alignment_study(stage, run_id=None):
    if stage not in PT100_STAGES:
        raise ValueError(f"unknown pt100-alignment stage: {stage}")
    run_id = run_id or pt100.PT100_ALIGNMENT_RUN
    paths = pt100.pt100_alignment_paths(PROJECT_CONFIG.data_root, TARGET_LAYER_DIR, run_id)
    workflow = full_workflow_manifest(paths["alignment"])
    inputs = {"digi_config": PROJECT_CONFIG.odd_digi_config,
              "nominal_geometry": data_path("odd_transforms.json")}
    if workflow is None:
        _require_completed(paths["simulation"])
        simulation, num_events = paths["simulation"], pt100.PT100_ALIGNMENT_EVENTS
        config = {"run_id": run_id, "num_events": num_events,
                  "source_run": pt100.PT100_SOURCE_RUN, "source_group": pt100.PT100_SOURCE_GROUP,
                  "pt_gev": 100.0, "fatras_interactions": True,
                  "misalignment_seed": pt100.PT100_MISALIGNMENT_SEED,
                  "misalignment": "per-sensor local tx/ty in +-[0.25,0.75] mm, rz in +-[0.01,0.03] rad",
                  "acts_max_iterations": pt100.PT100_ACTS_MAX_ITERATIONS,
                  "sequencer_threads": -1, "stage": stage}
        inputs["simulation_manifest"] = paths["simulation"] / "manifest.json"
    else:
        # A `full` run already holds its ACTS alignment and misaligned/aligned
        # reconstructions; only the Schur-geometry reconstruction is added here.
        if stage != "aligned-schur":
            raise ValueError(f"full-workflow run {run_id} only takes the aligned-schur stage")
        simulation = Path(workflow["outputs"]["simulation_dir"])
        num_events = workflow["metrics"]["workflow_config"]["num_events"]
        config = {**workflow["metrics"]["workflow_config"], "run_id": run_id,
                  "simulation_dir": str(simulation), "stage": stage}
        inputs["workflow_manifest"] = paths["alignment"] / "manifest.json"
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    if stage != "acts-alignment":
        _require_completed(paths["alignment"])
        inputs["misalignment"] = paths["misalignment"]
    aligned_file = None
    if stage == "aligned-acts":
        aligned_file = paths["acts_geometry"]
    elif stage == "aligned-schur":
        _require_completed(paths["schur_geometry"].parent)
        aligned_file = paths["schur_geometry"]
    if aligned_file is not None:
        inputs["aligned_geometry"] = aligned_file
    output = paths["alignment"] if stage == "acts-alignment" else paths[stage]
    detectors = []

    def prepare():
        # Every stage applies an alignment store, as HEAD's non-nominal modes did.
        detector = get_configured_open_data_detector(misaligned=True)
        detectors.append(detector)  # Keep detector elements alive through the run.
        common = dict(trackingGeometry=detector.trackingGeometry(), field=field,
                      digiConfigFile=PROJECT_CONFIG.odd_digi_config,
                      inputDir=simulation, detector=detector, numEvents=num_events)
        if stage == "acts-alignment":
            def align():
                # setupMisalignment draws from the module-level `random`.
                random.seed(pt100.PT100_MISALIGNMENT_SEED)
                runAlignment(**common, outputDir=output,
                             maxNumIterations=pt100.PT100_ACTS_MAX_ITERATIONS)
            return _DeferredRun(align)
        mode = "misaligned" if stage == "misaligned" else "aligned"
        suffix = "" if aligned_file is None else (
            f"{TARGET_LAYER_SUFFIX}_{stage.removeprefix('aligned-').replace('schur', 'schur_lm')}")
        return runReconstruction(**common, outputDir=output, mode=mode,
                                 alignmentDir=paths["alignment"], alignedFile=aligned_file,
                                 outputSuffix=suffix, finalOutputDir=True,
                                 overwriteOutputs=False)

    print(f"Starting pt100-alignment {stage} -> {output}", flush=True)
    run_material_study_stage(output, stage, prepare, inputs=inputs, config=config)
    if stage == "misaligned":
        link_pt100_nominal(paths)
    print(f"Completed pt100-alignment {stage}", flush=True)


def get_configured_open_data_detector(*, misaligned: bool):
    from acts.examples.odd import getOpenDataDetector

    return getOpenDataDetector(
        misaligned=misaligned,
        odd_dir=PROJECT_CONFIG.odd_root,
    )


def reconstruction_output_name(stem: str, output_suffix: str = "") -> str:
    if output_suffix:
        return f"{stem}_{output_suffix}.root"
    return artifact_filename(stem, ".root")


def reserve_reconstruction_output(
    output_dir: Path, stem: str, output_suffix: str, overwrite: bool
) -> Path:
    """Return an output path, refusing to let RECREATE clobber existing data.

    Every ACTS ROOT writer opens its file with fileMode="RECREATE", so reusing a
    suffix silently destroys a previous run's artifact.
    """
    destination = Path(output_dir) / reconstruction_output_name(stem, output_suffix)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing ROOT output: {destination}. "
            "Pass --overwrite or choose a different --output-suffix."
        )
    return destination


def parse_geoid(id_str):
    """Parse ID string, e.g. 'vol=16|lay=4|sen=1|ext=2'."""
    parts = {}
    for part in id_str.split("|"):
        key, value = part.split("=")
        parts[key] = int(value)
    return parts


def geoid_to_id_string(gid: acts.GeometryIdentifier) -> str:
    """Convert GeometryIdentifier to ID string format: 'vol=X|lay=Y|sen=Z|ext=W'."""
    parts = [f"vol={gid.volume}", f"lay={gid.layer}", f"sen={gid.sensitive}"]
    if gid.extra != 0:
        parts.append(f"ext={gid.extra}")
    return "|".join(parts)


def match_filter(geoid_parts, target_vol, target_lay, target_sen, target_ext):
    """
    Check if geoid matches filtering conditions.

    Note: Some elements don't have 'ext' parameter (e.g., volume=17,24,29)
    - If target_ext == -1: match all elements (with or without ext)
    - If target_ext != -1: only match elements that have ext AND ext == target_ext

    target_vol can be:
    - -1: match all volumes
    - int: match specific volume
    - list: match any volume in the list
    - dict: hierarchical structure {volume: {layer: [sensors]}}
        Example: {16: {4: [1, 2, 3], 5: [1, 2]}, 17: {2: [1]}}
        If layer value is -1 or list contains -1, match all sensors in that layer
        If layer value is a list, match sensors in any of those layers

    target_lay, target_sen: used for backward compatibility when target_vol is not a dict
    - -1: match all
    - int: match specific value
    - list: match any value in the list
    """
    vol_id = geoid_parts.get("vol")
    lay_id = geoid_parts.get("lay")
    sen_id = geoid_parts.get("sen")

    # Support hierarchical structure: {volume: {layer: [sensors]}}
    if isinstance(target_vol, dict):
        # Check if volume exists in config
        if vol_id not in target_vol:
            return False

        layer_config = target_vol[vol_id]

        # If layer_config is -1, match all layers and sensors in this volume
        if layer_config == -1:
            pass  # Continue to check ext
        elif isinstance(layer_config, dict):
            # Check if layer exists in config
            if lay_id not in layer_config:
                return False

            sensor_config = layer_config[lay_id]

            # If sensor_config is -1, match all sensors in this layer
            if sensor_config == -1:
                pass  # Continue to check ext
            elif isinstance(sensor_config, list):
                # Check if sensor is in the list
                if sen_id not in sensor_config:
                    return False
            else:
                # Single sensor value
                if sen_id != sensor_config:
                    return False
        elif isinstance(layer_config, list):
            # List of layers: match sensors in any of these layers
            if lay_id not in layer_config:
                return False
        else:
            # Single layer value
            if lay_id != layer_config:
                return False
    else:
        # Backward compatibility: original flat structure
        # Support target_vol as list or single value
        if target_vol != -1:
            if isinstance(target_vol, list):
                if vol_id not in target_vol:
                    return False
            else:
                if vol_id != target_vol:
                    return False

        # Support target_lay as list or single value
        if target_lay != -1:
            if isinstance(target_lay, list):
                if lay_id not in target_lay:
                    return False
            else:
                if lay_id != target_lay:
                    return False

        # Support target_sen as list or single value
        if target_sen != -1:
            if isinstance(target_sen, list):
                if sen_id not in target_sen:
                    return False
            else:
                if sen_id != target_sen:
                    return False

    if target_ext != -1:
        if "ext" not in geoid_parts or geoid_parts["ext"] != target_ext:
            return False
    return True


def match_surface_to_filter(gid, target_vol, target_lay, target_sen, target_ext):
    """
    Check if surface's GeoID matches filtering conditions

    target_vol can be:
    - -1: match all volumes
    - int: match specific volume
    - list: match any volume in the list
    - dict: hierarchical structure {volume: {layer: [sensors]}}
        Example: {16: {4: [1, 2, 3], 5: [1, 2]}, 17: {2: [1]}}
        If layer value is -1 or list contains -1, match all sensors in that layer
        If layer value is a list, match sensors in any of those layers

    target_lay, target_sen: used for backward compatibility when target_vol is not a dict
    - -1: match all
    - int: match specific value
    - list: match any value in the list
    """
    vol_id = gid.volume
    lay_id = gid.layer
    sen_id = gid.sensitive

    # Support hierarchical structure: {volume: {layer: [sensors]}}
    if isinstance(target_vol, dict):
        # Check if volume exists in config
        if vol_id not in target_vol:
            return False

        layer_config = target_vol[vol_id]

        # If layer_config is -1, match all layers and sensors in this volume
        if layer_config == -1:
            pass  # Continue to check ext
        elif isinstance(layer_config, dict):
            # Check if layer exists in config
            if lay_id not in layer_config:
                return False

            sensor_config = layer_config[lay_id]

            # If sensor_config is -1, match all sensors in this layer
            if sensor_config == -1:
                pass  # Continue to check ext
            elif isinstance(sensor_config, list):
                # Check if sensor is in the list
                if sen_id not in sensor_config:
                    return False
            else:
                # Single sensor value
                if sen_id != sensor_config:
                    return False
        elif isinstance(layer_config, list):
            # List of layers: match sensors in any of these layers
            if lay_id not in layer_config:
                return False
        else:
            # Single layer value
            if lay_id != layer_config:
                return False
    else:
        # Backward compatibility: original flat structure
        # Support target_vol as list or single value
        if target_vol != -1:
            if isinstance(target_vol, list):
                if vol_id not in target_vol:
                    return False
            else:
                if vol_id != target_vol:
                    return False

        # Support target_lay as list or single value
        if target_lay != -1:
            if isinstance(target_lay, list):
                if lay_id not in target_lay:
                    return False
            else:
                if lay_id != target_lay:
                    return False

        # Support target_sen as list or single value
        if target_sen != -1:
            if isinstance(target_sen, list):
                if sen_id not in target_sen:
                    return False
            else:
                if sen_id != target_sen:
                    return False

    if target_ext != -1:
        if gid.extra != target_ext:
            return False
    return True


# ============================================================================
# Simulation functions
# ============================================================================


def runSimulation(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    outputDir: Path,
    detector=None,
    numEvents=100,
    s: acts.examples.Sequencer = None,
    fixedPtGeV: Optional[float] = None,
    enableInteractions: bool = True,
):
    """
    Run simulation with misaligned geometry.
    """
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addDigitization,
    )

    s = s or acts.examples.Sequencer(
        events=numEvents, numThreads=-1, logLevel=acts.logging.INFO
    )

    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)
    outputDir.mkdir(exist_ok=True)

    logger = acts.logging.getLogger("Simulation")

    addParticleGun(
        s,
        ParticleConfig(num=1, pdg=acts.PdgParticle.eMuon, randomizeCharge=True),
        EtaConfig(-3.0, 3.0, uniform=True),
        MomentumConfig((1.0 if fixedPtGeV is None else fixedPtGeV) * u.GeV,
                       (100.0 if fixedPtGeV is None else fixedPtGeV) * u.GeV,
                       transverse=True),
        PhiConfig(0.0 * u.degree, 360.0 * u.degree),
        vtxGen=acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0),
            stddev=acts.Vector4(2.0 * u.mm, 2.0 * u.mm, 2.0 * u.mm, 2.0 * u.ns),
        ),
        multiplicity=1,
        rnd=rnd,
        outputDirCsv=outputDir,
        outputDirRoot=outputDir,
    )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        enableInteractions=enableInteractions,
        outputDirCsv=outputDir,
        outputDirRoot=outputDir,
    )

    digiAlg = addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        outputDirCsv=outputDir,
        outputDirRoot=outputDir,
    )

    # Note: particle_measurements_map will be generated after simulation
    # using the post-processing function below
    logger.info("Simulation output will be saved to: %s", outputDir)
    logger.info(
        "particle_measurements_map will be generated after simulation completes"
    )

    return s


# ============================================================================
# Alignment functions
# ============================================================================


def setupMisalignment(
    detector_elements: list,
    outputDir: Path,
    target_volume=-1,  # Can be int or list of ints
    target_layer=-1,  # Can be int or list of ints
    target_sensitive=-1,  # Can be int or list of ints
    target_extra: int = -1,
    tx: int = 0,
    ty: int = 0,
    tz: int = 1,
    rx: int = 0,
    ry: int = 0,
    rz: int = 0,
    shift_mag_mm: float = 2.0,
    rotation_mag_rad: float = 0.02,
):
    """
    Setup misalignment for detector elements

    Parameters
    ----------
    detector_elements : list
        List of all detector elements from odd_transforms.json
    outputDir : Path
        Directory to save misalignment record
    target_volume : int or list of ints
        Filter for volume IDs (-1 means no restriction, or list of specific IDs)
    target_layer : int or list of ints
        Filter for layer IDs (-1 means no restriction, or list of specific IDs)
    target_sensitive : int or list of ints
        Filter for sensitive IDs (-1 means no restriction, or list of specific IDs)
    target_extra : int
        Filter for extra ID (-1 means no restriction)
    tx, ty, tz : int
        Translation DOF flags (0 or 1)
    rx, ry, rz : int
        Rotation DOF flags (0 or 1)
    shift_mag_mm : float
        Random shift magnitude in mm
    rotation_mag_rad : float
        Random rotation magnitude in radians

    Returns
    -------
    tuple
        (geoIdMap, selected_elements, misalignment_record)
    """
    print("\n" + "=" * 60)
    print("Setting up misalignment in LOCAL coordinate system...")
    print("=" * 60)

    print(f"Misalignment DOF enabled (in local coordinates):")
    print(f"  Translation: tx={tx} (local X), ty={ty} (local Y), tz={tz} (local Z)")
    print(f"  Rotation:    rx={rx} (local X), ry={ry} (local Y), rz={rz} (local Z)")
    print(
        f"  Magnitudes:  ±{shift_mag_mm} mm (translation), ±{rotation_mag_rad} rad (rotation)"
    )
    print(f"\nNote: Using AlignmentGenerator LocalShift and LocalRotation:")
    print(f"  - LocalShift: Applies translation along specified local axis")
    print(f"  - LocalRotation: Applies rotation around specified local axis")
    print(f"  - Transformations are automatically handled in local coordinate system")

    # Filter elements to misalign
    selected_elements = []
    for element in detector_elements:
        geoid_parts = parse_geoid(element["ID"])
        if match_filter(
            geoid_parts, target_volume, target_layer, target_sensitive, target_extra
        ):
            selected_elements.append(element)

    print(f"\nFilter criteria:")

    # Format volume output (handle hierarchical dict, list, or single value)
    if isinstance(target_volume, dict):
        volume_str = "Hierarchical config:\n"
        for vol_id, layer_config in sorted(target_volume.items()):
            if layer_config == -1:
                volume_str += f"    Volume {vol_id}: ALL layers, ALL sensors\n"
            elif isinstance(layer_config, dict):
                volume_str += f"    Volume {vol_id}:\n"
                for lay_id, sensor_config in sorted(layer_config.items()):
                    if sensor_config == -1:
                        volume_str += f"      Layer {lay_id}: ALL sensors\n"
                    elif isinstance(sensor_config, list):
                        sensor_list = sensor_config[:10]
                        sensor_str = (
                            f"{sensor_list}{'...' if len(sensor_config) > 10 else ''}"
                        )
                        volume_str += f"      Layer {lay_id}: sensors {sensor_str}\n"
                    else:
                        volume_str += f"      Layer {lay_id}: sensor {sensor_config}\n"
            elif isinstance(layer_config, list):
                volume_str += (
                    f"    Volume {vol_id}: layers {layer_config}, ALL sensors\n"
                )
            else:
                volume_str += (
                    f"    Volume {vol_id}: layer {layer_config}, ALL sensors\n"
                )
        volume_str = volume_str.rstrip()
    elif target_volume == -1:
        volume_str = "ALL"
    elif isinstance(target_volume, list):
        volume_str = (
            f"{target_volume}"
            if len(target_volume) <= 5
            else f"{len(target_volume)} IDs: {target_volume[:5]}..."
        )
    else:
        volume_str = str(target_volume)
    print(f"  Volume:    {volume_str}")

    # Format layer output (only used for backward compatibility)
    if isinstance(target_volume, dict):
        layer_str = "N/A (using hierarchical config)"
    elif target_layer == -1:
        layer_str = "ALL"
    elif isinstance(target_layer, list):
        layer_str = (
            f"{target_layer}"
            if len(target_layer) <= 5
            else f"{len(target_layer)} IDs: {target_layer[:5]}..."
        )
    else:
        layer_str = str(target_layer)
    print(f"  Layer:     {layer_str}")

    # Format sensitive output (only used for backward compatibility)
    if isinstance(target_volume, dict):
        sensitive_str = "N/A (using hierarchical config)"
    elif target_sensitive == -1:
        sensitive_str = "ALL"
    elif isinstance(target_sensitive, list):
        sensitive_str = f'{len(target_sensitive)} IDs: {target_sensitive[:10]}{"..." if len(target_sensitive) > 10 else ""}'
    else:
        sensitive_str = str(target_sensitive)
    print(f"  Sensitive: {sensitive_str}")

    print(f"  Extra:     {target_extra if target_extra != -1 else 'ALL'}")
    print(f"Selected elements: {len(selected_elements)}")
    print(f"Shift magnitude: ±{shift_mag_mm} mm")
    print(f"Rotation magnitude: ±{rotation_mag_rad} rad")

    # Construct geoIdMap, apply random misalignment in local coordinate system
    # Also build misalignment record simultaneously
    # Using AlignmentGenerator LocalShift and LocalRotation for proper local transformations
    geoIdMap = {}
    misalignment_record = []

    # Random shift with asymmetric distribution
    # Each axis randomly chooses positive or negative side
    def asymmetric_random(mag, enabled):
        if not enabled:
            return 0.0
        if random.random() < 0.5:
            return random.uniform(-mag * 1.5, -mag * 0.5)
        else:
            return random.uniform(mag * 0.5, mag * 1.5)

    for element in selected_elements:
        geoid_parts = parse_geoid(element["ID"])
        nom_trans = element["Translation"]
        nom_rot_matrix = element["RotationMatrix"]

        # Create GeometryIdentifier
        gid_kwargs = {
            "volume": geoid_parts.get("vol", 0),
            "layer": geoid_parts.get("lay", 0),
            "sensitive": geoid_parts.get("sen", 0),
        }
        if "ext" in geoid_parts:
            gid_kwargs["extra"] = geoid_parts["ext"]
        gid = acts.GeometryIdentifier(**gid_kwargs)

        # Read rotation matrix directly from JSON
        R_nominal = np.array(nom_rot_matrix)

        # Create nominal Transform3
        nom_rotation = acts.RotationMatrix3(
            acts.Vector3(R_nominal[0, 0], R_nominal[1, 0], R_nominal[2, 0]),
            acts.Vector3(R_nominal[0, 1], R_nominal[1, 1], R_nominal[2, 1]),
            acts.Vector3(R_nominal[0, 2], R_nominal[1, 2], R_nominal[2, 2]),
        )
        trf = acts.Transform3(
            acts.Vector3(nom_trans[0], nom_trans[1], nom_trans[2]), nom_rotation
        )

        # Store local shift and rotation values for recording
        shift_local_x = asymmetric_random(shift_mag_mm, tx != 0)
        shift_local_y = asymmetric_random(shift_mag_mm, ty != 0)
        shift_local_z = 0
        rot_local_z = asymmetric_random(rotation_mag_rad, rz != 0)
        rot_local_y = 0
        rot_local_x = 0

        # ===== Apply LocalShift using AlignmentGenerator =====
        # LocalShift applies translation in local coordinate system
        if tx != 0:
            lShiftX = acts.examples.alignment.AlignmentGeneratorLocalShift()
            lShiftX.axisDirection = acts.AxisDirection.AxisX
            lShiftX.shift = shift_local_x
            lShiftX(trf)  # Apply local shift along X axis

        if ty != 0:
            lShiftY = acts.examples.alignment.AlignmentGeneratorLocalShift()
            lShiftY.axisDirection = acts.AxisDirection.AxisY
            lShiftY.shift = shift_local_y
            lShiftY(trf)  # Apply local shift along Y axis

        # ===== Apply LocalRotation using AlignmentGenerator =====
        # LocalRotation applies rotation around local axes
        # The axis parameter is expressed in the transform's local coordinates because
        # AlignmentGeneratorLocalRotation applies the rotation by right multiplication.

        if rz != 0:
            lRotZ = acts.examples.alignment.AlignmentGeneratorLocalRotation()
            lRotZ.axis = acts.Vector3(0.0, 0.0, 1.0)
            lRotZ.angle = rot_local_z
            lRotZ(trf)  # Apply rotation around local Z axis

        # Store the misaligned transform
        geoIdMap[gid] = trf

        # Extract misaligned transform components
        misaligned_trans = trf.translation
        misaligned_rot = trf.rotation

        # Extract rotation matrix as nested list
        rotation_matrix = [
            [
                float(misaligned_rot[0, 0]),
                float(misaligned_rot[0, 1]),
                float(misaligned_rot[0, 2]),
            ],
            [
                float(misaligned_rot[1, 0]),
                float(misaligned_rot[1, 1]),
                float(misaligned_rot[1, 2]),
            ],
            [
                float(misaligned_rot[2, 0]),
                float(misaligned_rot[2, 1]),
                float(misaligned_rot[2, 2]),
            ],
        ]

        # Simplified record: only ID, Translation, and RotationMatrix
        record = {
            "ID": element["ID"],
            "Translation": [
                misaligned_trans[0],
                misaligned_trans[1],
                misaligned_trans[2],
            ],
            "RotationMatrix": rotation_matrix,
        }
        misalignment_record.append(record)

    print(f"\nTotal misaligned detector elements: {len(geoIdMap)}")

    # Save misalignment record
    misalignment_file = outputDir / artifact_filename("misalignment_applied", ".json")
    with open(misalignment_file, "w") as f:
        json.dump(misalignment_record, f, indent=2)
    print(f"Misalignment record saved to: {misalignment_file}")

    return geoIdMap, selected_elements, misalignment_record


def runAlignment(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    inputDir: Path,
    outputDir: Path,
    detector=None,
    numEvents=100,
    reverseFilteringMomThreshold=0 * u.GeV,
    reverseFilteringCovarianceScaling=1,
    s: acts.examples.Sequencer = None,
    maxNumIterations: int = 100,
):
    from acts.examples.simulation import (
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        addKalmanTracks,
    )

    s = s or acts.examples.Sequencer(
        events=numEvents, numThreads=-1, logLevel=acts.logging.INFO
    )

    rnd = acts.examples.RandomNumbers(seed=42)
    inputDir = Path(inputDir)
    outputDir = target_subdir(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    logger = acts.logging.getLogger("Alignment")

    # Load odd_transforms.json
    json_path = data_path("odd_transforms.json")
    print(f"Loading detector transforms from: {json_path}")

    with open(json_path, "r") as f:
        detector_elements = json.load(f)
    print(f"Total detector elements in JSON: {len(detector_elements)}")

    # Setup misalignment parameters
    # Format 1: Hierarchical structure - specify volume -> layer -> sensors
    # Example: {16: {4: [1, 2, 3], 5: [1, 2]}, 17: {2: [1]}}
    #   - Volume 16, Layer 4: sensors [1, 2, 3]
    #   - Volume 16, Layer 5: sensors [1, 2]
    #   - Volume 17, Layer 2: sensor [1]
    # Use -1 for layer value to match all sensors in that layer
    # Use -1 for sensor list to match all sensors: {16: {4: -1}} means all sensors in vol16/lay4
    # Format 1: Use hierarchical structure (dict) - specify volume -> layer -> sensors
    target_volume = target_volume_map()
    # Format 2: Use flat structure (backward compatible)
    # target_volume = -1      # -1 = all volumes, or dict for hierarchical: {vol: {lay: [sensors]}}
    # target_layer = -1        # -1 = all layers (used only if target_volume is not a dict)
    # target_sensitive = -1   # Used only if target_volume is not a dict
    # target_extra = -1       # -1 = all extra

    # These variables must be defined even when using hierarchical structure (for backward compatibility)
    target_layer = -1  # -1 = all layers (used only if target_volume is not a dict)
    target_sensitive = -1  # Used only if target_volume is not a dict
    target_extra = -1  # -1 = all extra

    initialvarinflation = [100, 100, 100, 100, 100, 1]

    tx, ty, tz = 1, 1, 0  # Translation DOF
    rx, ry, rz = 0, 0, 1  # Rotation DOF
    shift_mag_mm = 0.5  # Shift magnitude (mm)
    rotation_mag_rad = 0.02  # Rotation magnitude (rad)

    geoIdMap, selected_elements, misalignment_record = setupMisalignment(
        detector_elements=detector_elements,
        outputDir=outputDir,
        target_volume=target_volume,
        target_layer=target_layer,
        target_sensitive=target_sensitive,
        target_extra=target_extra,
        tx=tx,
        ty=ty,
        tz=tz,
        rx=rx,
        ry=ry,
        rz=rz,
        shift_mag_mm=shift_mag_mm,
        rotation_mag_rad=rotation_mag_rad,
    )
    mutableStore = acts.examples.alignment.MutableGeoIdAlignmentStore(geoIdMap)

    # Configure AlignmentDecorator (IOV can be set to global interval)
    cfg = AlignmentDecorator.Config()
    cfg.iovStores = [((0, 1_000_000), mutableStore)]  # Effective for all events
    cfg.garbageCollection = False
    alignDeco = AlignmentDecorator(cfg, acts.logging.INFO)
    s.addContextDecorator(alignDeco)

    logger.info("Reading simulation results from CSV files in %s", inputDir)
    s.addReader(
        acts.examples.CsvSimHitReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            inputStem="hits",
            outputSimHits="simhits",
        )
    )

    # Read measurements from CSV
    s.addReader(
        acts.examples.CsvMeasurementReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            outputMeasurements="measurements",
            outputMeasurementSimHitsMap="measurement_simhits_map",
            outputMeasurementParticlesMap="measurement_particles_map",
            outputParticleMeasurementsMap="particle_measurements_map",
            inputSimHits="simhits",
        )
    )

    # Read simulated particles from CSV
    s.addReader(
        acts.examples.CsvParticleReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            inputStem="particles_simulated",
            outputParticles="particles_simulated",
        )
    )

    s.addWhiteboardAlias("particles", "particles_simulated")
    s.addWhiteboardAlias("particles_simulated_selected", "particles_simulated")
    s.addWhiteboardAlias("particles_selected", "particles_simulated")

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(2 * u.GeV, None),
            measurements=(5, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_selected",
        initialVarInflation=initialvarinflation,
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        particleHypothesis=acts.ParticleHypothesis.muon,
    )

    print("\n" + "=" * 60)
    print("Setting up alignment algorithm...")
    print("=" * 60)

    # Alignment DOF masks (should match misalignment DOF)
    Center0 = 1 << 0  # dx (translation X) - matches tx
    Center1 = 1 << 1  # dy (translation Y) - matches ty
    Center2 = 1 << 2  # dz (translation Z) - matches tz
    Rotation0 = 1 << 3  # rx (rotation around X) - matches rx
    Rotation1 = 1 << 4  # ry (rotation around Y) - matches ry
    Rotation2 = 1 << 5  # rz (rotation around Z) - matches rz

    # Enable alignment DOF based on misalignment DOF
    alignment_dof = 0
    if tx:
        alignment_dof |= Center0
    if ty:
        alignment_dof |= Center1
    if tz:
        alignment_dof |= Center2
    if rx:
        alignment_dof |= Rotation0
    if ry:
        alignment_dof |= Rotation1
    if rz:
        alignment_dof |= Rotation2

    print(f"Alignment DOF enabled (matches misalignment):")
    print(
        f"  dx={bool(alignment_dof & Center0)}, dy={bool(alignment_dof & Center1)}, dz={bool(alignment_dof & Center2)}"
    )
    print(
        f"  rx={bool(alignment_dof & Rotation0)}, ry={bool(alignment_dof & Rotation1)}, rz={bool(alignment_dof & Rotation2)}"
    )
    print(f"  Alignment mask value: {alignment_dof} (binary: {bin(alignment_dof)})")

    # Collect detector elements to align (same as misaligned elements)
    # Use the same filtering logic as misalignment section
    print("\nCollecting detector elements for alignment...")
    print(f"Using same filter as misalignment:")
    if isinstance(target_volume, dict):
        print(f"  Volume:    Hierarchical config (see misalignment section above)")
    else:
        print(f"  Volume:    {target_volume if target_volume != -1 else 'ALL'}")
    if not isinstance(target_volume, dict):
        print(f"  Layer:     {target_layer if target_layer != -1 else 'ALL'}")
        print(f"  Sensitive: {target_sensitive if target_sensitive != -1 else 'ALL'}")
    print(f"  Extra:     {target_extra if target_extra != -1 else 'ALL'}")

    det_elements = []

    def visit(surface: acts.Surface) -> bool:
        # Use surfacePlacement instead of associatedDetectorElement
        # because DetectorElementBase is not registered in Python bindings
        placement = acts.examples.alignment.surfacePlacement(surface)
        if placement is not None:
            gid = surface.geometryId
            # Use same filter as misalignment
            if match_surface_to_filter(
                gid, target_volume, target_layer, target_sensitive, target_extra
            ):
                det_elements.append(placement)
        return True

    trackingGeometry.visitSurfaces(visit)

    print(f"Total detector elements selected for alignment: {len(det_elements)}")

    # Verify the count matches misaligned elements
    if len(det_elements) != len(geoIdMap):
        print(
            f"WARNING: Mismatch between misaligned ({len(geoIdMap)}) and alignment ({len(det_elements)}) element counts!"
        )
    else:
        print(f"Alignment element count matches misaligned elements")

    # Configure AlignmentAlgorithm
    aal_cfg = acts.examples.alignment.AlignmentAlgorithmConfig()
    aal_cfg.inputMeasurements = "measurements"
    aal_cfg.inputProtoTracks = "truth_particle_tracks"
    aal_cfg.inputInitialTrackParameters = "estimatedparameters"
    aal_cfg.outputAlignmentParameters = "alignmentParameters"

    # Set trackingGeometry (for surfaceAccessor)
    aal_cfg.trackingGeometry = trackingGeometry
    aal_cfg.chi2ONdfCutOff = 0.1
    aal_cfg.deltaChi2ONdfCutOff = (5, 0.0001)

    # Create AlignedTransformUpdater, it will update mutableStore after each iteration
    aal_cfg.alignedTransformUpdater = acts.examples.alignment.makeAlignedTransformUpdater(
        mutableStore
    )

    aal_cfg.alignedDetElements = det_elements
    aal_cfg.align = acts.examples.alignment.makeAlignmentFunction(
        trackingGeometry, field, acts.logging.INFO
    )
    aal_cfg.maxNumIterations = maxNumIterations
    aal_cfg.maxNumTracks = 100000
    aal_cfg.iterationState = {
        i: int(alignment_dof) for i in range(aal_cfg.maxNumIterations)
    }

    alignment_algo = acts.examples.alignment.AlignmentAlgorithm(aal_cfg, acts.logging.INFO)
    s.addAlgorithm(alignment_algo)

    # Run the sequencer
    s.run()

    print("\n" + "=" * 60)
    print("Extracting aligned transforms...")
    print("=" * 60)

    # Extract aligned transforms from mutableStore after alignment
    aligned_record = []
    transform_map = mutableStore.getTransformMap()
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()

    # Collect surfaces that were aligned (same filter as misalignment)
    aligned_surfaces = []

    def visit(surface: acts.Surface) -> bool:
        # Use surfacePlacement to check if surface has a placement
        placement = acts.examples.alignment.surfacePlacement(surface)
        if placement is not None:
            gid = surface.geometryId
            # Use same filter as misalignment
            if match_surface_to_filter(
                gid, target_volume, target_layer, target_sensitive, target_extra
            ):
                aligned_surfaces.append(surface)
        return True

    trackingGeometry.visitSurfaces(visit)

    for surface in aligned_surfaces:
        gid = surface.geometryId
        element_id = geoid_to_id_string(gid)

        # Find the matching GeometryIdentifier in transform_map
        aligned_trf = None
        for map_gid, trf in transform_map.items():
            if str(map_gid) == str(gid):
                aligned_trf = trf
                break

        if aligned_trf is not None:
            aligned_trans = aligned_trf.translation
            aligned_rot = aligned_trf.rotation

            # Extract aligned rotation matrix as nested list
            aligned_rotation_matrix = [
                [
                    float(aligned_rot[0, 0]),
                    float(aligned_rot[0, 1]),
                    float(aligned_rot[0, 2]),
                ],
                [
                    float(aligned_rot[1, 0]),
                    float(aligned_rot[1, 1]),
                    float(aligned_rot[1, 2]),
                ],
                [
                    float(aligned_rot[2, 0]),
                    float(aligned_rot[2, 1]),
                    float(aligned_rot[2, 2]),
                ],
            ]

            # Simplified record: only ID, Translation, and RotationMatrix
            record = {
                "ID": element_id,
                "Translation": [aligned_trans[0], aligned_trans[1], aligned_trans[2]],
                "RotationMatrix": aligned_rotation_matrix,
            }
            aligned_record.append(record)
        else:
            print(f"Warning: Could not find aligned transform for {element_id}")

    # Save aligned transforms record
    aligned_file = outputDir / artifact_filename("aligned_transforms", ".json")
    with open(aligned_file, "w") as f:
        json.dump(aligned_record, f, indent=2)
    print(f"\nAligned transforms saved to: {aligned_file}")
    print(f"Total aligned elements: {len(aligned_record)}")

    # Summary information
    print(f"\nTotal aligned elements: {len(aligned_record)}")

    print("\n" + "=" * 60)
    print("Alignment completed successfully!")
    print("=" * 60)
    print(f"\nOutput files saved to: {outputDir}")
    print("\nGenerated files:")
    print(f"  - {artifact_filename('misalignment_applied', '.json')}: Applied misalignment transforms")
    print("    * Includes Translation and RotationMatrix")
    print(f"  - {artifact_filename('aligned_transforms', '.json')}: Final aligned transforms")
    print("    * Includes Translation and RotationMatrix")

    return s


# ============================================================================
# Reconstruction functions
# ============================================================================


def runReconstruction(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    inputDir: Path,
    outputDir: Path,
    detector=None,
    numEvents=10,
    reverseFilteringMomThreshold=0 * u.GeV,
    reverseFilteringCovarianceScaling=1,
    s: acts.examples.Sequencer = None,
    mode: str = "nominal",  # "nominal", "misaligned", or "aligned"
    alignmentDir: Path = None,  # Directory containing alignment output files
    alignedFile: str = None,  # Override path for the aligned transforms JSON
    outputSuffix: str = "",  # Appended as _{suffix} to output ROOT file names
    recordMaterial: bool = False,  # Also record per-track material interactions
    recordSteps: bool = False,  # Keep the stepping logger alive for diagnostics
    materialEnergyLoss: bool = False,  # Gate 0 ablation only; see below
    overwriteOutputs: bool = False,  # Allow RECREATE over an existing ROOT file
    finalOutputDir: bool = False,  # Internal study path already includes target/group
):
    """
    Run reconstruction with different geometry modes.

    Parameters
    ----------
    mode : str
        - "nominal": Use nominal geometry (no alignment decorator)
        - "misaligned": Use misaligned geometry (loads misalignment_applied_<suffix>.json)
        - "aligned": Use aligned geometry (loads aligned_transforms_<suffix>.json)
        The <suffix> and the target-layer subdirectory both come from
        alignment_artifact_layout; do not hardcode layer numbers here.
    alignmentDir : Path
        Directory containing alignment output files (default: alignment_output)
    """
    from acts.examples.simulation import (
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        addKalmanTracks,
    )

    s = s or acts.examples.Sequencer(
        events=numEvents, numThreads=-1, logLevel=acts.logging.INFO
    )

    rnd = acts.examples.RandomNumbers(seed=42)
    inputDir = Path(inputDir)
    outputDir = Path(outputDir) if finalOutputDir else target_subdir(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    logger = acts.logging.getLogger("Reconstruction")

    # Determine alignment directory
    if alignmentDir is None:
        alignmentDir = target_subdir(data_path("alignment_output"))
    else:
        alignmentDir = target_subdir(alignmentDir)

    # Load transforms based on mode
    if mode == "nominal":
        print("=" * 60)
        print("Running reconstruction with NOMINAL geometry...")
        print("=" * 60)
        # No alignment decorator needed for nominal geometry
        use_alignment_decorator = False
    elif mode == "misaligned":
        print("=" * 60)
        print("Running reconstruction with MISALIGNED geometry...")
        print("=" * 60)
        json_path = alignmentDir / artifact_filename("misalignment_applied", ".json")
        print(f"Loading misaligned transforms from: {json_path}")
        use_alignment_decorator = True
    elif mode == "aligned":
        print("=" * 60)
        print("Running reconstruction with ALIGNED geometry...")
        print("=" * 60)
        if alignedFile is not None:
            json_path = resolve_project_path(alignedFile)
        else:
            json_path = alignmentDir / artifact_filename("aligned_transforms", ".json")
        print(f"Loading aligned transforms from: {json_path}")
        use_alignment_decorator = True
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'nominal', 'misaligned', or 'aligned'"
        )

    if use_alignment_decorator:
        if not json_path.exists():
            print(f"Error: Transform file {json_path} does not exist!")
            print(f"Please run alignment mode first to generate the required files.")
            sys.exit(1)

        with open(json_path, "r") as f:
            detector_elements = json.load(f)

        print(f"Total detector elements in JSON: {len(detector_elements)}")
        print(f"Using all elements: {len(detector_elements)}")
        print(f"Loading transforms from JSON file...")
        print()

        geoIdMap = {}
        for idx, element in enumerate(detector_elements):
            geoid_parts = parse_geoid(element["ID"])

            gid_kwargs = {
                "volume": geoid_parts.get("vol", 0),
                "layer": geoid_parts.get("lay", 0),
                "sensitive": geoid_parts.get("sen", 0),
            }
            if "ext" in geoid_parts:
                gid_kwargs["extra"] = geoid_parts["ext"]
            gid = acts.GeometryIdentifier(**gid_kwargs)

            # Use transform data directly from JSON (simplified structure)
            trans_data = element["Translation"]
            rot_matrix = element["RotationMatrix"]

            trans = acts.Vector3(trans_data[0], trans_data[1], trans_data[2])

            R = np.array(rot_matrix)
            rotation = acts.RotationMatrix3(
                acts.Vector3(R[0, 0], R[1, 0], R[2, 0]),
                acts.Vector3(R[0, 1], R[1, 1], R[2, 1]),
                acts.Vector3(R[0, 2], R[1, 2], R[2, 2]),
            )

            trf = acts.Transform3(trans, rotation)
            geoIdMap[gid] = trf

        print(f"\nTotal detector elements loaded: {len(geoIdMap)}")
        mutableStore = acts.examples.alignment.MutableGeoIdAlignmentStore(geoIdMap)

        cfg = AlignmentDecorator.Config()
        cfg.iovStores = [((0, 1_000_000), mutableStore)]  # Effective for all events
        cfg.garbageCollection = False
        alignDeco = AlignmentDecorator(cfg, acts.logging.INFO)

        s.addContextDecorator(alignDeco)
    else:
        print("Using nominal geometry (no alignment decorator)")

    logger.info("Reading particles from %s", inputDir / "particles.root")

    s.addReader(
        acts.examples.CsvSimHitReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            inputStem="hits",
            outputSimHits="simhits",
        )
    )

    s.addReader(
        acts.examples.CsvMeasurementReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            outputMeasurements="measurements",
            outputMeasurementSimHitsMap="measurement_simhits_map",
            outputMeasurementParticlesMap="measurement_particles_map",
            outputParticleMeasurementsMap="particle_measurements_map",
            inputSimHits="simhits",
        )
    )

    s.addReader(
        acts.examples.CsvParticleReader(
            level=acts.logging.INFO,
            inputDir=str(inputDir),
            inputStem="particles_simulated",
            outputParticles="particles_simulated",
        )
    )

    s.addWhiteboardAlias("particles", "particles_simulated")
    s.addWhiteboardAlias("particles_simulated_selected", "particles_simulated")
    s.addWhiteboardAlias("particles_selected", "particles_simulated")
    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(2 * u.GeV, None),
            measurements=(5, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_selected",
        initialVarInflation=[1] * 6,
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        particleHypothesis=acts.ParticleHypothesis.muon,
    )

    addKalmanTracks(
        s,
        trackingGeometry,
        field,
        reverseFilteringMomThreshold,
        reverseFilteringCovarianceScaling,
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks="tracks",
            outputTracks="selected-tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=5,
            ),
        )
    )
    s.addWhiteboardAlias("tracks", "selected-tracks")

    if recordMaterial:
        # Material export shares this exact chain -- same geometry, field, seed,
        # particle selection, fitter and track selection -- so the recorded nodes
        # belong to the very tracks written by RootTrackStatesWriter below.
        #
        # RootMaterialTrackWriter drops the material map key and iterates an
        # unordered_map (RootMaterialTrackWriter.cpp:87), so the material rows
        # carry no track index.  RootPropagationSummaryWriter is the index
        # bridge: it emits (event_nr, track_nr) alongside the same start bound
        # parameters that PropagationAlgorithm handed to the material recorder.
        #
        # energyLoss/multipleScattering stay off on purpose.  The first
        # broken-helix version keeps curvature constant between segments and
        # does not model energy loss; letting q/p evolve here would record nodes
        # along a trajectory the fitted model cannot represent.
        s.addAlgorithm(
            acts.examples.TracksToParameters(
                level=acts.logging.INFO,
                inputTracks="tracks",
                outputTrackParameters="params-from-tracks",
            )
        )
        propagator = acts.examples.ConcretePropagator(
            acts.Propagator(
                acts.EigenStepper(field),
                acts.Navigator(trackingGeometry=trackingGeometry),
            )
        )
        s.addAlgorithm(
            acts.examples.PropagationAlgorithm(
                level=acts.logging.INFO,
                propagatorImpl=propagator,
                # A sterile SteppingLogger empties PropagationSummary::steps, which
                # is what RootPropagationSummaryWriter counts nMaterials from
                # (RootPropagationSummaryWriter.cpp:141-151).  Keeping it sterile
                # is cheap for production; Gate 0 turns it on so nMaterials becomes
                # a real cross-check of the recorded surface-material nodes.
                sterileLogger=not recordSteps,
                recordMaterialInteractions=True,
                energyLoss=materialEnergyLoss,
                multipleScattering=False,
                covarianceTransport=False,
                inputTrackParameters="params-from-tracks",
                outputSummaryCollection="propagation_summary",
                outputMaterialCollection="material_tracks",
            )
        )
        s.addWriter(
            acts.examples.root.RootMaterialTrackWriter(
                level=acts.logging.INFO,
                inputMaterialTracks="material_tracks",
                filePath=str(
                    reserve_reconstruction_output(
                        outputDir, "material_kf", outputSuffix, overwriteOutputs
                    )
                ),
                storeSurface=True,
                storeVolume=True,
                prePostStep=False,
                collapseInteractions=False,
            )
        )
        s.addWriter(
            acts.examples.root.RootPropagationSummaryWriter(
                level=acts.logging.INFO,
                inputSummaryCollection="propagation_summary",
                filePath=str(
                    reserve_reconstruction_output(
                        outputDir, "propsummary_kf", outputSuffix, overwriteOutputs
                    )
                ),
            )
        )

    s.addWriter(
        acts.examples.root.RootTrackStatesWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            inputSimHits="simhits",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(
                reserve_reconstruction_output(
                    outputDir, "trackstates_kf", outputSuffix, overwriteOutputs
                )
            ),
        )
    )

    s.addWriter(
        acts.examples.root.RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(
                reserve_reconstruction_output(
                    outputDir, "tracksummary_kf", outputSuffix, overwriteOutputs
                )
            ),
        )
    )

    s.addWriter(
        acts.examples.root.RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(
                reserve_reconstruction_output(
                    outputDir, "performance_kf", outputSuffix, overwriteOutputs
                )
            ),
        )
    )

    return s


# ============================================================================
# Main entry point
# ============================================================================

if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="Complete simulation, alignment, and reconstruction workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run complete workflow (simulation -> alignment -> reconstruction with all geometries)
            python alignment_odd.py full

            # Run individual steps
            python alignment_odd.py simulation
            python alignment_odd.py alignment
            python alignment_odd.py reconstruction --mode nominal
            python alignment_odd.py reconstruction --mode misaligned
            python alignment_odd.py reconstruction --mode aligned
        """,
    )
    add_config_argument(parser)

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Full workflow command
    full_parser = subparsers.add_parser(
        "full",
        help="Run complete workflow: simulation -> alignment -> reconstruction (all geometries)",
    )
    full_parser.add_argument(
        "--num-events", type=int, default=100000, help="Number of events"
    )
    full_parser.add_argument(
        "--simulation-dir",
        type=absolute_project_path,
        default=None,
        help="Simulation output directory (default: simulation_output)",
    )
    full_parser.add_argument(
        "--alignment-dir",
        type=absolute_project_path,
        default=None,
        help="Alignment output directory (default: alignment_output)",
    )
    full_parser.add_argument(
        "--reconstruction-dir",
        type=absolute_project_path,
        default=None,
        help="Reconstruction output directory (default: reconstruction_output)",
    )
    full_parser.add_argument(
        "--pt-gev",
        type=float,
        default=None,
        help="Fixed simulated muon pT in GeV (default: uniform 1-100 GeV)",
    )

    # Simulation command
    sim_parser = subparsers.add_parser("simulation", help="Run simulation")
    sim_parser.add_argument(
        "--num-events", type=int, default=100000, help="Number of events"
    )
    sim_parser.add_argument(
        "--output-dir",
        type=absolute_project_path,
        default=None,
        help="Output directory (default: simulation_output)",
    )

    # Alignment command
    align_parser = subparsers.add_parser("alignment", help="Run alignment algorithm")
    align_parser.add_argument(
        "--num-events", type=int, default=100000, help="Number of events"
    )
    align_parser.add_argument(
        "--input-dir",
        type=absolute_project_path,
        default=None,
        help="Input directory (default: simulation_output)",
    )
    align_parser.add_argument(
        "--output-dir",
        type=absolute_project_path,
        default=None,
        help="Output directory (default: alignment_output)",
    )

    # Reconstruction command
    recon_parser = subparsers.add_parser("reconstruction", help="Run reconstruction")
    recon_parser.add_argument(
        "--mode",
        type=str,
        choices=["nominal", "misaligned", "aligned"],
        default="nominal",
        help="Reconstruction mode",
    )
    recon_parser.add_argument(
        "--num-events", type=int, default=100000, help="Number of events"
    )
    recon_parser.add_argument(
        "--input-dir",
        type=absolute_project_path,
        default=None,
        help="Input directory (default: simulation_output)",
    )
    recon_parser.add_argument(
        "--output-dir",
        type=absolute_project_path,
        default=None,
        help="Output directory (default: reconstruction_output)",
    )
    recon_parser.add_argument(
        "--aligned-file",
        type=absolute_project_path,
        default=None,
        help="Path to aligned transforms JSON (for --mode aligned; overrides default path)",
    )
    recon_parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output ROOT file names (e.g. 'ml' -> trackstates_kf_ml.root)",
    )

    # Material export command
    material_parser = subparsers.add_parser(
        "material",
        help="Reconstruct and additionally record per-track material interactions",
    )
    material_parser.add_argument(
        "--num-events", type=int, default=100000, help="Number of events"
    )
    material_parser.add_argument(
        "--input-dir",
        type=absolute_project_path,
        default=None,
        help="Input directory (default: simulation_output)",
    )
    material_parser.add_argument(
        "--output-dir",
        type=absolute_project_path,
        default=None,
        help="Output directory (default: reconstruction_output)",
    )
    material_parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output ROOT file names (e.g. 'mat0')",
    )
    material_parser.add_argument(
        "--run-tag",
        type=str,
        required=True,
        help="Manifest run identifier for this material export",
    )
    material_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow RECREATE to replace existing ROOT outputs",
    )
    material_parser.add_argument(
        "--energy-loss",
        action="store_true",
        help=(
            "Gate 0 ablation only: let q/p evolve during the recording pass. "
            "Production keeps this off so the recorded nodes stay consistent "
            "with the constant-curvature broken-helix model."
        ),
    )
    material_parser.add_argument(
        "--record-steps",
        action="store_true",
        help=(
            "Keep the stepping logger alive so nMaterials is populated. "
            "Costs memory per track; intended for Gate 0 diagnostics."
        ),
    )
    material_parser.add_argument(
        "--capture-source-snapshot",
        action="store_true",
        help="Archive the exact tracked+untracked source tree into the run's "
        "manifest so it can be replayed independent of git commit/dirty state.",
    )

    pt100_parser = subparsers.add_parser(
        "pt100-alignment",
        help="Pinned pT=100 GeV material-on ACTS-Alignment vs Schur-LM study stage",
    )
    pt100_parser.add_argument("stage", choices=PT100_STAGES)
    pt100_parser.add_argument(
        "--run-id", default=None,
        help="Run under data/alignment/ (default: the pinned study run); a completed "
        "`full` run takes only aligned-schur",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup common paths and detector
    digiConfigFile = PROJECT_CONFIG.odd_digi_config
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    # Execute commands
    if args.command == "full":
        # Full workflow: simulation -> alignment -> reconstruction (all three geometries)
        print("\n" + "=" * 80)
        print("=" * 80)
        print("RUNNING COMPLETE WORKFLOW")
        print("=" * 80)
        print("=" * 80)

        simulationDir = (
            Path(args.simulation_dir)
            if args.simulation_dir
            else data_path("simulation_output")
        )
        alignmentDir = target_subdir(
            Path(args.alignment_dir)
            if args.alignment_dir
            else data_path("alignment_output")
        )
        reconstructionDir = (
            Path(args.reconstruction_dir)
            if args.reconstruction_dir
            else data_path("reconstruction_output")
        )

        manifest_path = alignmentDir / "manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to reuse full-workflow run: {manifest_path}")
        manifest = write_experiment_manifest(
            alignmentDir,
            run_id=f"full/{alignmentDir.parent.name}",
            command=[sys.executable, *sys.argv],
            config=PROJECT_CONFIG,
            inputs={
                "digi_config": digiConfigFile,
                "nominal_geometry": data_path("odd_transforms.json"),
            },
            outputs={
                "simulation_dir": simulationDir,
                "alignment_dir": alignmentDir,
                "reconstruction_dir": reconstructionDir,
            },
            metrics={
                "workflow_config": {
                    "num_events": args.num_events,
                    "pt_gev": args.pt_gev,
                    "fatras_interactions": True,
                    "sequencer_threads": -1,
                    "random_seed": 42,
                    "field_tesla": 2.0,
                }
            },
            status="running",
            capture_source_snapshot=True,
        )
        try:
            # Step 1: Simulation (NOMINAL geometry — controlled experiment)
            # Hits are generated at nominal positions (clean baseline).
            # Misalignment is injected only into the reconstruction geometry (Step 2+),
            # so we know the ground truth and can measure how well alignment recovers it.
            print("\n" + "=" * 80)
            print("STEP 1: SIMULATION (using NOMINAL geometry)")
            print("=" * 80)
            detector_nom = get_configured_open_data_detector(misaligned=False)
            trackingGeometry_nom = detector_nom.trackingGeometry()

            runSimulation(
                trackingGeometry=trackingGeometry_nom,
                field=field,
                digiConfigFile=digiConfigFile,
                outputDir=simulationDir,
                detector=detector_nom,
                numEvents=args.num_events,
                fixedPtGeV=args.pt_gev,
            ).run()

            # Step 2: Alignment (starting from MISALIGNED geometry)
            # The alignment algorithm starts from the perturbed geometry and uses the
            # nominal hits from Step 1 to learn corrections back toward nominal.
            print("\n" + "=" * 80)
            print("STEP 2: ALIGNMENT (starting from MISALIGNED geometry)")
            print("=" * 80)
            alignmentDir.mkdir(parents=True, exist_ok=True)

            detector_mis = get_configured_open_data_detector(misaligned=True)
            trackingGeometry_mis = detector_mis.trackingGeometry()

            runAlignment(
                trackingGeometry=trackingGeometry_mis,
                field=field,
                digiConfigFile=digiConfigFile,
                inputDir=simulationDir,
                outputDir=alignmentDir,
                detector=detector_mis,
                numEvents=args.num_events,
            )

            # Step 3: Reconstruction with nominal geometry
            print("\n" + "=" * 80)
            print("STEP 3: RECONSTRUCTION WITH NOMINAL GEOMETRY")
            print("=" * 80)
            # detector_nom and trackingGeometry_nom already created in Step 1
            reconDir_nom = target_subdir(reconstructionDir / "nominal")
            reconDir_nom.mkdir(parents=True, exist_ok=True)

            runReconstruction(
                trackingGeometry=trackingGeometry_nom,
                field=field,
                digiConfigFile=digiConfigFile,
                inputDir=simulationDir,
                outputDir=reconDir_nom,
                detector=detector_nom,
                numEvents=args.num_events,
                mode="nominal",
                alignmentDir=alignmentDir,
            ).run()

            # Step 4: Reconstruction with misaligned geometry
            print("\n" + "=" * 80)
            print("STEP 4: RECONSTRUCTION WITH MISALIGNED GEOMETRY")
            print("=" * 80)
            reconDir_mis = target_subdir(reconstructionDir / "misaligned")
            reconDir_mis.mkdir(parents=True, exist_ok=True)

            runReconstruction(
                trackingGeometry=trackingGeometry_mis,
                field=field,
                digiConfigFile=digiConfigFile,
                inputDir=simulationDir,
                outputDir=reconDir_mis,
                detector=detector_mis,
                numEvents=args.num_events,
                mode="misaligned",
                alignmentDir=alignmentDir,
            ).run()

            # Step 5: Reconstruction with aligned geometry
            print("\n" + "=" * 80)
            print("STEP 5: RECONSTRUCTION WITH ALIGNED GEOMETRY")
            print("=" * 80)
            reconDir_ali = target_subdir(reconstructionDir / "aligned")
            reconDir_ali.mkdir(parents=True, exist_ok=True)

            runReconstruction(
                trackingGeometry=trackingGeometry_mis,
                field=field,
                digiConfigFile=digiConfigFile,
                inputDir=simulationDir,
                outputDir=reconDir_ali,
                detector=detector_mis,
                numEvents=args.num_events,
                mode="aligned",
                alignmentDir=alignmentDir,
            ).run()
        except BaseException as error:
            finalize_experiment_manifest(
                manifest,
                status="failed",
                metrics={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        finalize_experiment_manifest(
            manifest,
            status="completed",
            outputs={
                "aligned_transforms": alignmentDir
                / artifact_filename("aligned_transforms", ".json"),
                "misalignment_applied": alignmentDir
                / artifact_filename("misalignment_applied", ".json"),
                "reconstruction_nominal": reconDir_nom,
                "reconstruction_misaligned": reconDir_mis,
                "reconstruction_aligned": reconDir_ali,
            },
        )

        print("\n" + "=" * 80)
        print("=" * 80)
        print("COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
        print("=" * 80)
        print("=" * 80)
        print(f"\nOutput directories:")
        print(f"  - Simulation: {simulationDir}")
        print(f"  - Alignment: {alignmentDir}")
        print(f"  - Reconstruction (nominal): {reconDir_nom}")
        print(f"  - Reconstruction (misaligned): {reconDir_mis}")
        print(f"  - Reconstruction (aligned): {reconDir_ali}")

    elif args.command == "simulation":
        run_material_study("simulation")

    elif args.command == "alignment":
        inputDir = (
            Path(args.input_dir) if args.input_dir else data_path("simulation_output")
        )
        outputDir = (
            Path(args.output_dir)
            if args.output_dir
            else data_path("alignment_output")
        )
        outputDir.mkdir(parents=True, exist_ok=True)

        if not inputDir.exists():
            print(f"Error: Input directory {inputDir} does not exist!")
            print("Please run simulation first to generate simulation data.")
            sys.exit(1)

        detector = get_configured_open_data_detector(misaligned=True)
        trackingGeometry = detector.trackingGeometry()

        runAlignment(
            trackingGeometry=trackingGeometry,
            field=field,
            digiConfigFile=digiConfigFile,
            inputDir=inputDir,
            outputDir=outputDir,
            detector=detector,
            numEvents=args.num_events,
        )

    elif args.command == "reconstruction":
        run_material_study("reconstruction")

    elif args.command == "pt100-alignment":
        run_pt100_alignment_study(args.stage, args.run_id)

    elif args.command == "material":
        inputDir = (
            Path(args.input_dir) if args.input_dir else data_path("simulation_output")
        )
        baseoutputDir = (
            Path(args.output_dir)
            if args.output_dir
            else data_path("reconstruction_output")
        )
        outputDir = target_subdir(baseoutputDir / "nominal")
        outputDir.mkdir(parents=True, exist_ok=True)

        if not inputDir.exists():
            print(f"Error: Input directory {inputDir} does not exist!")
            print("Please run simulation first to generate simulation data.")
            sys.exit(1)

        # Material export is itself an algorithm experiment: the manifest is
        # created before the Sequencer starts so an abort still leaves a record.
        # Export and conversion are separate experiments; giving each its own
        # run directory is what keeps both manifests from colliding.
        manifest_dir = (
            target_subdir(data_path("material_output")) / args.run_tag / "export"
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        material_root = outputDir / reconstruction_output_name(
            "material_kf", args.output_suffix
        )
        propsummary_root = outputDir / reconstruction_output_name(
            "propsummary_kf", args.output_suffix
        )
        manifest = write_experiment_manifest(
            manifest_dir,
            run_id=args.run_tag,
            command=[sys.executable, *sys.argv],
            config=load_project_config(),
            inputs={"simulation_dir": inputDir},
            outputs={
                "material_root": material_root,
                "propsummary_root": propsummary_root,
                "trackstates_root": outputDir
                / reconstruction_output_name("trackstates_kf", args.output_suffix),
                "tracksummary_root": outputDir
                / reconstruction_output_name("tracksummary_kf", args.output_suffix),
            },
            metrics={
                "exporter_config": {
                    "stepper": "EigenStepper",
                    "navigator": "Navigator",
                    "energy_loss": args.energy_loss,
                    "multiple_scattering": False,
                    "covariance_transport": False,
                    "record_material_interactions": True,
                    "sterile_logger": not args.record_steps,
                    "store_surface": True,
                    "store_volume": True,
                    "pre_post_step": False,
                    "collapse_interactions": False,
                    "num_events": args.num_events,
                    "random_seed": 42,
                    "particle_hypothesis": "muon",
                    "min_measurements": 5,
                    "pt_min_gev": 2.0,
                }
            },
            status="running",
            capture_source_snapshot=args.capture_source_snapshot,
        )

        detector = get_configured_open_data_detector(misaligned=False)
        trackingGeometry = detector.trackingGeometry()
        try:
            runReconstruction(
                trackingGeometry=trackingGeometry,
                field=field,
                digiConfigFile=digiConfigFile,
                inputDir=inputDir,
                outputDir=outputDir,
                detector=detector,
                numEvents=args.num_events,
                mode="nominal",
                outputSuffix=args.output_suffix,
                recordMaterial=True,
                recordSteps=args.record_steps,
                materialEnergyLoss=args.energy_loss,
                overwriteOutputs=args.overwrite,
            ).run()
        except BaseException as error:
            finalize_experiment_manifest(
                manifest,
                status="failed",
                metrics={"error": f"{type(error).__name__}: {error}"},
            )
            raise
        finalize_experiment_manifest(manifest, status="completed")

        print("\n" + "=" * 60)
        print("Material export completed successfully!")
        print("=" * 60)
        print(f"\nOutput files saved to: {outputDir}")
        print(f"  - {material_root.name}: material interactions")
        print(f"  - {propsummary_root.name}: propagation summary (track index bridge)")
        print(f"\nManifest: {manifest}")
