# Paired helix-recovery diagnostic

Run this diagnostic in the conda `pytorch` environment. Input and target-layer
defaults come from the project configuration, including the target-derived ROOT
and geometry filenames. Use a fresh output directory for every retained run.

Quick smoke run:

```bash
python scripts/compare_helix_recovery.py \
  --event-partition train \
  --max-tracks 25 \
  --seed-iters 20 \
  --refit-iters 10 \
  --bootstrap-replicates 100 \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/smoke
```

Calibration run for inspecting the frozen diagnostic and its independent
protected/exposed verdicts:

```bash
python scripts/compare_helix_recovery.py \
  --event-partition calibration \
  --exposure both \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/calibration
```

Validation run with the production defaults (200 seed iterations, 60 refit
iterations, and 2000 event-cluster bootstrap replicates):

```bash
python scripts/compare_helix_recovery.py \
  --event-partition validation \
  --exposure both \
  --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/validation
```

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
