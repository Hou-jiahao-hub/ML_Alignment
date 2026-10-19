# Input and output path contract

Two path conventions apply:

- **`mlalign` package** (`mlalign-align`, `mlalign.run_alignment`): every input
  and output path is explicit; relative paths resolve against the caller's
  working directory. The package has no default data paths.
- **Workflow scripts** (`scripts/`): relative configuration values and CLI
  arguments resolve from the repository root, never from the caller's working
  directory. Print the effective absolute roots with:

```bash
python scripts/alignment_artifact_layout.py --shell
```

## Current machine roots

| Purpose | Effective absolute path |
|---|---|
| Repository | `/home/houjiahao/ML_Alignment` |
| Local artifacts | `/home/houjiahao/ML_Alignment/data` |
| ACTS checkout | `/home/houjiahao/alignTest` |
| ODD checkout | `/home/houjiahao/alignTest/thirdparty/OpenDataDetector` |
| Conda shell hook | `/home/houjiahao/miniconda3/etc/profile.d/conda.sh` |

These values are examples of the current effective configuration, not constants
embedded in source code. Machine-specific values belong in the ignored
`configs/local.toml`.

## External inputs

| Input | Default path | Consumer |
|---|---|---|
| ACTS environment setup | `<acts_root>/setup_acts.sh` | `run_schur_experiment.sh` |
| ODD material map | `<odd_root>/data/odd-material-maps.root` | ODD geometry construction |
| Digitization config | `<acts_root>/Examples/Configs/odd-digi-smearing-config.json` | Simulation and reconstruction |
| Conda environment setup | `$HOME/miniconda3/etc/profile.d/conda.sh` | `run_schur_experiment.sh` |

## Data inputs and outputs

`<target>` is `Vol17_Lay8__Vol24_Lay4` and `<suffix>` is `v17l8_v24l4` for the
versioned default configuration.

| Stage | Inputs | Outputs |
|---|---|---|
| Geometry export | ODD geometry and material map | `data/geometry/odd_transforms.json` |
| Simulation | ODD geometry and digi configuration | `data/simulation/particles.root` plus `event#########-{particles,particles_simulated,hits,cells,measurements,measurement-simhit-map}.csv` |
| Misalignment + ACTS-Alignment | Nominal geometry and simulation artifacts | `data/alignment/<target>/misalignment_applied_<suffix>.json` and `aligned_transforms_<suffix>.json` (ACTS baseline) |
| Reconstruction | Simulation artifacts and selected geometry JSON | `data/reconstruction/{nominal,misaligned,aligned}/<target>/{trackstates,tracksummary,performance}_kf_<suffix-or-run>.root` |
| Schur-LM (`mlalign-align`) | Input geometry JSON, its trackstates ROOT, full nominal geometry JSON | `--output-geom` JSON (input target surfaces), optional `--output-summary` JSON, optional `loss_curve_*.png` with `--plot` |
| Residual diagnostics | Reconstruction ROOT files | `data/reconstruction/residuals/<target>/*.root` |
| Chi2 diagnostics | Tracksummary ROOT files | `data/reconstruction/chi2_ndf/<target>/*.png` |
| Schur-LM experiment | Misaligned geometry, trackstates/tracksummary, nominal trackstates | `data/alignment/<target>/aligned_transforms_<run_tag>_schur_lm.json`, `data/reconstruction/aligned/<target>/*_kf_<run_tag>_schur_lm.root`, and acceptance, summaries, plots, logs and `manifest.json` under `data/experiments/schur_lm/<target>/<run_tag>` |

## CLI and failure behavior

- Absolute arguments are normalized without changing their target.
- Workflow: relative arguments such as `inputs/states.root` resolve to
  `<repository_root>/inputs/states.root` from every working directory.
- `mlalign-align`: relative arguments resolve against the working directory.
- Missing inputs report their path at the point of use.
- Output commands create parent directories only when executing the operation.
- Existing run-tag namespaces remain protected from accidental overwrite.
