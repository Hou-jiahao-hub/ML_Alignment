# Experiment artifact cleanup — 2026-09-02

## Scope and safety boundary

This record inventories large, reproducible experiment outputs before any
deletion.  The repository configuration currently targets
`Vol17_Lay8__Vol24_Lay4` (`alignment.target_layers = ["17:8", "24:4"]`).

The cleanup must retain:

- all nominal and misaligned reconstruction inputs;
- `data/simulation/particles.root` and geometry inputs;
- every alignment geometry JSON, especially
  `aligned_transforms_trial-008_joint_free.json`;
- every experiment `manifest.json`, compact summary JSON, plot, and historical
  result note;
- the trial-008 aligned reconstruction and the 2026-09-01 joint-only outputs.

Only derived aligned-reconstruction ROOT outputs are candidates below.  No
source ROOT/JSON artifact is in scope.  Deletion requires a final exact-path
inventory and explicit user approval.

## Storage snapshot

Snapshot date: 2026-09-02 (Asia/Shanghai).

- `data/`: approximately 122 GiB as reported by `du`.
- `data/reconstruction/aligned/`: approximately 101 GiB.
- Filesystem free space before cleanup: approximately 720 GiB.
- Candidate derived outputs: 101,211,291,358 bytes (approximately 94.3 GiB).

## Candidate batch A — superseded target-layer combinations

These directories are not selected by the current configuration.  The latest
handoff explicitly states that the accepted conclusion has not been reproduced
on these combinations, so their aligned ROOT files do not support the accepted
trial-008 claim.

| Directory | Trackstates | Tracksummary | Performance | Total bytes |
|---|---:|---:|---:|---:|
| `data/reconstruction/aligned/Vol16_Lay6__Vol23_Lay4/` | 35 | 35 | 35 | 34,575,569,942 |
| `data/reconstruction/aligned/Vol17_Lay2__Vol24_Lay6/` | 11 | 11 | 11 | 12,421,025,120 |

Batch A total: 46,996,595,062 bytes (approximately 43.8 GiB).

Relevant result point retained: these runs are historical exploration only;
the final free-helix acceptance result is explicitly limited to
`Vol17_Lay8__Vol24_Lay4` and was not reproduced on either combination.
Their small geometry JSON files and non-aligned nominal/misaligned inputs remain
available for a future reproduction.

## Candidate batch B — superseded current-target reconstructions

The following filename families live only under
`data/reconstruction/aligned/Vol17_Lay8__Vol24_Lay4/`.  The corresponding
geometry JSON, manifests, compact summaries, plots, and history entries are
retained.

| Family | Files | Bytes | Retained result point |
|---|---:|---:|---|
| `*autograd*` | 30 | 11,289,984,237 | Early joint/free optimization established the viable coupled-fit direction; later mainline trials supersede these reconstructions. |
| `*fresh_surface_helix*` | 21 | 7,911,380,478 | Warm-start exploration was an intermediate implementation stage, not the accepted endpoint. |
| `*trial-001*` | 21 | 7,902,828,567 | Early surface-only iteration; superseded by the joint mainline and retained only as compact records/geometry. |
| `*trial-004*` | 21 | 7,902,888,924 | Early surface-only iteration; superseded by later solver and acceptance changes. |
| `*trial-005*` | 21 | 7,902,836,860 | Early surface-only iteration; superseded by trial-006/007/008 investigation. |
| `*trial-006*` | 15 | 5,648,498,247 | Joint update was ineffective: the joint output matched warm2, and all four acceptance components failed. |
| `*trial-007*` | 27 | 5,656,278,983 | First full mainline run failed blind validation (only 24:4 LOC1 passed); the Adam measurement refit did not converge and joint geometry was rejected. |

Batch B total: 54,214,696,296 bytes (approximately 50.5 GiB).

The decisive replacement is trial-008.  With identical inputs, event split,
threshold profile, and sample counts, changing the measurement refit from Adam
to Gauss--Newton reduced refit chi2/component from 7595.638 to 0.763146.  All
four blind-validation components passed, with normalized values 0.7042,
0.8445, 0.8851, and 0.9600.  The accepted geometry is
`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_trial-008_joint_free.json`.

## Explicitly deferred from cleanup

- `data/reconstruction/aligned/Vol17_Lay8__Vol24_Lay4/*trial-008*`
- `data/reconstruction/aligned/Vol17_Lay8__Vol24_Lay4/*jointonly*`
- current default/base aligned outputs not matching a candidate family
- all `data/experiments/**` artifacts, including large helix-validation CSVs
- all nominal, misaligned, simulation, geometry, alignment JSON, manifest,
  summary, and plot artifacts

The large helix-validation CSVs can be evaluated as a separate cleanup batch
after the aligned ROOT cleanup is verified; mixing them into this first batch
would make provenance review less clear.

## Required deletion-time checks

1. Generate the exact candidate path list and total byte count without globs in
   the destructive command.
2. Confirm every target is a regular file below
   `data/reconstruction/aligned/` and none is a symlink.
3. Re-run `scripts/path_inventory.py --json` and confirm no active input or
   default output is targeted.
4. Obtain explicit approval for batches A and/or B.
5. Delete only the approved exact paths, then re-run `du`, path inventory, and
   the quick test suite.
6. Append actual deleted counts/bytes and verification results to this record.

## Execution and verification

Executed on 2026-09-02 after explicit approval for batches A+B.

- Deleted files: 294 regular derived ROOT files.
- Deleted bytes: 101,211,291,358 (approximately 94.3 GiB).
- Removed empty batch-A directories:
  `Vol16_Lay6__Vol23_Lay4` and `Vol17_Lay2__Vol24_Lay6` below
  `data/reconstruction/aligned/`.
- Candidate files remaining after deletion: 0.
- `data/` after cleanup: approximately 28 GiB (`du -sh`).
- `data/reconstruction/aligned/` after cleanup: approximately 6.5 GiB.
- Filesystem free space after cleanup: approximately 815 GiB.

Post-delete protection checks confirmed that the accepted trial-008 geometry
and aligned ROOT outputs, the 2026-09-01 joint-only outputs, current nominal and
misaligned trackstates, and `data/simulation/particles.root` remain present.
`scripts/path_inventory.py --json` still resolves the active target to
`Vol17_Lay8__Vol24_Lay4` and reports the expected active inputs and output
directories.

Quick-suite verification in the conda `pytorch` environment:

```text
351 passed, 3 skipped, 11 deselected in 107.77s
```

The three skips are environment-sensitive checks: one requires the optional
AlignDerivLib pybind extension and two require ACTS Python bindings.  No test
failed.
