# Paired helix-recovery diagnostic

Run this diagnostic in the conda `pytorch` environment. Input and target-layer
defaults come from the project configuration, including the target-derived ROOT
and geometry filenames. Use a fresh output directory for every retained run.

Quick smoke run over the lowest 25 unique event numbers:

```bash
python scripts/compare_helix_recovery.py \
  --max-events 25 \
  --seed-iters 20 \
  --refit-iters 10 \
  --bootstrap-replicates 100 \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/smoke-25-events
```

The retained comparison uses one deterministic 20,000-event cohort. After full
nominal-versus-misaligned pairing, the loader keeps every track and hit belonging
to the lowest 20,000 unique `event_nr` values. The manifest records the available
and selected unique-event counts. The removed event-partition CLI is intentionally
not accepted; use `--max-events` and `--max-tracks` for bounded runs.

Full four-arm validation:

```bash
python scripts/compare_helix_recovery.py \
  --max-events 20000 \
  --exposure both \
  --seed-iters 200 \
  --refit-iters 60 \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 6006 \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/validation-20000-events
```

## Deterministic sampled four-arm validation

For a full paired input whose target-hit probe rows are too numerous for a
practical CPU four-arm solve, sample after strict pairing and frozen-probe
construction, but before any seed or refit.  The sampling unit is a probe row;
it is not a bootstrap draw.  Within each target layer and prediction mode, a
fixed SHA-256 ordering selects at most the requested number of rows, at most a
fixed number of tracks per event.  Reusing the input and sampling seed therefore
reuses exactly the same cohort.

Smoke the sampling contract with 100 rows per stratum:

```bash
python scripts/compare_helix_recovery.py \
  --max-events 0 --max-tracks 0 --exposure both \
  --sample-rows-per-stratum 100 --sampling-seed 6006 \
  --max-probes-per-event-stratum 2 --analysis-modes interpolation \
  --seed-iters 20 --refit-iters 10 --bootstrap-replicates 100 \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/smoke-sampled-100
```

The retained production protocol uses 2,000 rows per available stratum:

```bash
python scripts/compare_helix_recovery.py \
  --max-events 0 --max-tracks 0 --exposure both \
  --sample-rows-per-stratum 2000 --sampling-seed 6006 \
  --max-probes-per-event-stratum 2 --analysis-modes interpolation \
  --seed-iters 600 --refit-iters 60 \
  --bootstrap-replicates 2000 --bootstrap-seed 6006 \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/full-sampled-interpolation
```

`prediction_mode_coverage` in the manifest and summary reports candidate and
selected rows, tracks, events, and pre-fit LOC availability for every requested
target-layer/mode stratum. Modes absent or underpowered in the input remain
diagnostic `inconclusive` groups; when `--analysis-modes interpolation` is used,
they do not enter the primary verdict and the result must not be extrapolated to
first-hit or last-hit targets. Bootstrap remains event-clustered: each draw
resamples whole events and retains every probe row selected from that event.

`failed` and `inconclusive` scientific verdicts still exit successfully and are
recorded as completed experiments. ROOT, geometry, plotting, numerical, or
keyboard-interrupt failures return nonzero and finalize the run manifest with
`status: failed`, retaining any cohort or scientific metrics already computed.

The summary always enumerates every requested target layer, prediction mode,
and local component. A requested layer with no retained probes is therefore
reported as `inconclusive`; if the whole selected cohort is empty, no seed or
refit solver is run and the completed manifest records the same verdict.

Before creating the output directory or manifest, the CLI rejects any resolved
path collision between source inputs (including an explicit configuration) and
its fixed output artifacts. Such an invalid invocation writes nothing, so it
cannot overwrite a source ROOT, geometry, or configuration file.
