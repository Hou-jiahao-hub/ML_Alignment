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
recorded as completed experiments. ROOT, geometry, plotting, or numerical
exceptions return nonzero and finalize the run manifest with `status: failed`.
