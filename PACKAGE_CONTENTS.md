# Package contents and evidence status

## Complete implementation

- `stg_ddqn/`: simulator, candidate generation, migration mechanics,
  graph-temporal Double DQN, IM, DLAPM, statistics, validation, and plotting.
- `configs/`: default, smoke, synthetic behavior-validation, and final SUMO
  paper profiles.
- `tests/`: resource, routing, migration, reward, reproducibility, result, and
  curve-only plotting regression tests.
- `tools/`: SUMO trace export and topology CSV generation.
- `scripts/`: smoke and behavior-validation workflows.
- `matlab/`: curve-only MATLAB figure generation from the Python summary CSV.
- `docs/`: equation mapping, baseline specification, experiment protocol,
  manuscript synchronization, and validation notes.

## Included measured diagnostic artifacts

- `checkpoints/behavior_stg_ddqn.pt`: compact behavior-validation checkpoint.
- `results/behavior/`: 135 synthetic paired-seed runs and request records.
- `results/behavior_training/`: training log for the diagnostic checkpoint.
- `plots/curve_only_behavior/`: combined and separate PDF/PNG figures with
  STG-DDQN, IM, and DLAPM curves and no shaded bands.

## Required before final manuscript claims

The package intentionally does not contain a fabricated SUMO trace or final
paper results. Add the frozen 1-second SUMO CSV at
`data/mobility/sumo_trace.csv`, run `configs/paper.yaml`, validate the ten-seed
outputs, and regenerate the figures. Until that run is completed, the included
curves are implementation diagnostics rather than publishable SUMO evidence.
