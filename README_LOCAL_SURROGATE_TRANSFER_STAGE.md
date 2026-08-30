# Source Local-Surrogate Transfer Pilot v1

This stage follows the source-local-structure extraction study and asks one narrower question:

> Under a correct, externally fixed source-target region correspondence, does the extracted source local relative-rank surrogate add held-out target prediction and candidate-ranking value beyond a target-only local GP?

## Scope

This is a controlled two-dimensional static held-out Pilot. It deliberately excludes:

- unknown region matching;
- learned coordinate alignment;
- full-covariance cross-task alignment;
- acquisition functions and candidate filtering;
- sequential Bayesian optimization;
- claims of general negative-transfer safety.

The source expert is associated with the target region using generator metadata. The source oracle center selects the nearest extracted structure; that structure is queried around its own extracted center and mapped by translation plus one frozen scalar chart radius to the oracle target center. This prevents source-model extrapolation from being confused with transfer failure. The extracted-to-oracle source-center error remains a diagnostic. The extracted full covariance is not treated as a transferable alignment estimate because the preceding recovery study did not establish superior covariance-shape recovery.

## Model conditions

All target GP conditions share the same fixed Matern-5/2 kernel and target observations.

- `Target-Only`
- `Source-Affine-Only`
- `Fixed-Source+Residual`
- `Calibrated-Source+Residual`
- `Gated-Source+Residual`

The calibrated model learns a non-negative, ridge-shrunk target-side coefficient for source cost `1 - h_s`. A non-positive association falls back exactly to Target-Only. The gated model additionally requires target-context cross-validation improvement and pairwise ordering agreement; rejected experts also fall back exactly to Target-Only.

Relations are `matching`, a realistic benchmark-specific `wrong` source, and an explicit `reversed` ordering counterfactual. The reversed expert is a safety stress test, not a natural task-distribution claim.

## Frozen protocol and configurations

- Protocol: `PROTOCOL_LOCAL_SURROGATE_TRANSFER_PILOT.md`
- Quick configuration: `configs/local_surrogate_transfer_quick.json`
- Full Pilot configuration: `configs/local_surrogate_transfer_full.json`

Quick results validate code paths only. Confirmatory interpretation uses the full configuration and its own output directory.

## Tests

```bash
python -m pytest tests/test_local_surrogate_transfer.py -q
python -m pytest -q
```

## Quick run

```bash
python scripts/run_all_local_surrogate_transfer_pilot.py \
  --config configs/local_surrogate_transfer_quick.json \
  --output results/local_surrogate_transfer_pilot_quick
```

## Full Pilot run

```bash
python scripts/run_all_local_surrogate_transfer_pilot.py \
  --config configs/local_surrogate_transfer_full.json \
  --output results/local_surrogate_transfer_pilot
```

## Expected artifacts

```text
results/local_surrogate_transfer_pilot/
├── local_surrogate_transfer_results.csv
├── local_surrogate_transfer_diagnostics.csv
├── local_surrogate_transfer_target_ledger.csv
├── local_surrogate_transfer_failures.csv
├── local_surrogate_transfer_manifest.json
├── AUDIT.json
├── FULL_RUN_AUDIT.md
└── analysis/
    ├── LOCAL_SURROGATE_TRANSFER_REPORT.md
    ├── local_surrogate_transfer_primary_tests.csv
    ├── local_surrogate_transfer_summary.csv
    ├── local_surrogate_transfer_gate_summary.csv
    ├── local_surrogate_transfer_matching_curves.png
    ├── local_surrogate_transfer_wrong_curves.png
    ├── local_surrogate_transfer_reversed_curves.png
    └── local_surrogate_transfer_gate.png
```

## Interpretation boundary

Source-expert fidelity, matching transfer benefit, wrong-source behavior, reversal safety, gate coverage, and risk among accepted instances are separate quantities. A supported matching result under oracle correspondence does not establish unknown-alignment transferability or online optimization benefit. A rejected source is not proven intrinsically non-transferable, and observed fallback safety is not a universal no-harm guarantee.
