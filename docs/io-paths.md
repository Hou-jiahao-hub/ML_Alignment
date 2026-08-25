# Input and output path contract

All paths are normalized before I/O. Relative configuration values and relative
CLI arguments resolve from the repository root, never from the caller's current
working directory. Run the authoritative inventory with:

```bash
python scripts/path_inventory.py --json
```

## Current machine roots

| Purpose | Effective absolute path |
|---|---|
| Repository | `/home/houjiahao/ML_Alignment` |
| Local artifacts | `/home/houjiahao/ML_Alignment/data` |
| ACTS checkout | `/home/houjiahao/alignTest` |
| AlignDerivLib | `/home/houjiahao/ML_Alignment/AlignDerivLib` |
| ODD checkout | `/home/houjiahao/alignTest/thirdparty/OpenDataDetector` |
| Conda shell hook | `/home/houjiahao/miniconda3/etc/profile.d/conda.sh` |

These values are examples of the current effective configuration, not constants
embedded in source code. Machine-specific values belong in the ignored
`configs/local.toml`.

## External inputs

| Input | Default path | Consumer |
|---|---|---|
| ACTS environment setup | `<acts_root>/setup_acts.sh` | Shell experiment drivers |
| ODD material map | `<odd_root>/data/odd-material-maps.root` | ODD geometry construction |
| Digitization config | `<acts_root>/Examples/Configs/odd-digi-smearing-config.json` | Simulation and reconstruction |
| pybind build directory | `<align_derivlib_root>/build_pybind11` | Analytic-Jacobian solver |
| Conda environment setup | `$HOME/miniconda3/etc/profile.d/conda.sh` | Shell experiment drivers |

Every placeholder root above is emitted as an absolute path by
`path_inventory.py`.

## Data inputs and outputs

`<target>` is `Vol17_Lay8__Vol24_Lay4` and `<suffix>` is `v17l8_v24l4` for the
versioned default configuration.

| Stage | Inputs | Outputs |
|---|---|---|
| Geometry export | ODD geometry and material map | `data/geometry/odd_transforms.json` |
| Simulation | ODD geometry and digi configuration | `data/simulation/particles.root` plus `event#########-{particles,particles_simulated,hits,cells,measurements,measurement-simhit-map}.csv` |
| Misalignment | Nominal geometry and simulation artifacts | `data/alignment/<target>/misalignment_applied_<suffix>.json` and `aligned_transforms_<suffix>.json` |
| Reconstruction | Simulation artifacts and selected geometry JSON | `data/reconstruction/{nominal,misaligned,aligned}/<target>/{trackstates,tracksummary,performance}_kf_<suffix-or-run>.root` |
| ML alignment | Geometry JSON, trackstates ROOT, tracksummary ROOT, nominal geometry | Updated geometry JSON and a sibling `loss_curve_*.png` |
| Residual diagnostics | Reconstruction ROOT files | `data/reconstruction/residuals/<target>/*.root` |
| Chi2/hit diagnostics | Tracksummary ROOT files | `data/reconstruction/{chi2_ndf,hit_number}/<target>/*.png` |
| Iterative experiment | Misaligned or prior-iteration geometry and ROOT files | Geometry, aligned reconstruction ROOT files, plots, and logs under `data/experiments/iterative_ml/<target>` |
| Surface/helix experiment | Nominal and candidate ROOT files plus geometry JSON | Run-tagged geometry, summaries, ROOT files, and logs under `data/experiments/surface_helix/<target>/<run_tag>` |
| Helix validation | Nominal trackstates and geometry JSON | `helix_validation_summary.json`, `helix_measurement_comparison.csv`, compatibility `helix_smt_comparison.csv`, `manifest.json`, and `plots/*.png` under `data/experiments/helix_validation/<target>` |

The inventory command publishes concrete default paths and absolute templates
containing `{run_tag}`, `{iteration}`, `{axis}`, or `{layer}` for dynamic files.

## CLI and failure behavior

- Absolute arguments are normalized without changing their target.
- Relative arguments such as `inputs/states.root` resolve to
  `<repository_root>/inputs/states.root` from every working directory.
- Missing inputs report their absolute path at the point of use.
- Output commands create parent directories only when executing the operation.
- Existing run-tag namespaces remain protected from accidental overwrite.
