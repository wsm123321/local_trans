# Target-proposal / source-region-screening research report

This report is generated from experiment CSV files; conclusions are not hardcoded.

## 1. Shared-proposal mechanism study

All methods used the same target observations, target GP, raw pool, and target-proposed candidates.

| Method | Mean normalized regret | Top-10% hit rate | Mean retained fraction | Filter activation |
|---|---:|---:|---:|---:|
| Target-Only | 0.7837 | 25.0% | 1.000 | 0.0% |
| Matching-Fixed-Filter | 0.3675 | 50.0% | 0.250 | 100.0% |
| Matching-Adaptive-Filter | 0.7171 | 25.0% | 0.856 | 100.0% |
| Matching-Soft-Rerank | 0.3996 | 50.0% | 1.000 | 0.0% |
| Random-Adaptive-Filter | 0.7837 | 25.0% | 0.938 | 100.0% |
| Wrong-Adaptive-Filter | 0.7135 | 25.0% | 0.869 | 100.0% |
| Oracle-Fixed-Filter | 0.5000 | 25.0% | 0.250 | 100.0% |

### Paired comparisons against Target-Only

- **Matching-Fixed-Filter**: regret reduction +0.4162 [+0.1404, +0.6920]; Wilcoxon p=0.25; paired win rate=75.0%.
- **Matching-Adaptive-Filter**: regret reduction +0.0665 [+0.0000, +0.1996]; Wilcoxon p=1; paired win rate=25.0%.
- **Matching-Soft-Rerank**: regret reduction +0.3841 [+0.0951, +0.6850]; Wilcoxon p=0.25; paired win rate=75.0%.
- **Random-Adaptive-Filter**: regret reduction +0.0000 [+0.0000, +0.0000]; Wilcoxon p=1; paired win rate=0.0%.
- **Wrong-Adaptive-Filter**: regret reduction +0.0702 [+0.0000, +0.2105]; Wilcoxon p=1; paired win rate=25.0%.
- **Oracle-Fixed-Filter**: regret reduction +0.2837 [+0.0665, +0.5620]; Wilcoxon p=0.25; paired win rate=75.0%.

## 2. Equal-budget sequential optimization

| Method | Final normalized regret | Total improvement | Filter activation | Mean trust |
|---|---:|---:|---:|---:|
| Target-Only | 0.5370 | 2.1502 | 0.0% | 0.000 |
| Matching-Fixed-Filter | 0.4648 | 2.2379 | 100.0% | 1.000 |
| Matching-Adaptive-Filter | 0.4948 | 2.1870 | 100.0% | 0.255 |
| Matching-Soft-Rerank | 0.5830 | 1.7730 | 0.0% | nan |
| Random-Adaptive-Filter | 0.5370 | 2.1502 | 95.0% | 0.099 |
| Wrong-Adaptive-Filter | 0.4873 | 2.2100 | 100.0% | 0.205 |
| Oracle-Fixed-Filter | 0.6086 | 1.7643 | 95.0% | 1.000 |

### Final-regret paired comparisons

- **Matching-Fixed-Filter**: final-regret reduction +0.0721 [+0.0000, +0.2008]; Wilcoxon p=0.5.
- **Matching-Adaptive-Filter**: final-regret reduction +0.0422 [+0.0000, +0.1265]; Wilcoxon p=1.
- **Matching-Soft-Rerank**: final-regret reduction -0.0461 [-0.1432, +0.0730]; Wilcoxon p=0.625.
- **Random-Adaptive-Filter**: final-regret reduction +0.0000 [+0.0000, +0.0000]; Wilcoxon p=1.
- **Wrong-Adaptive-Filter**: final-regret reduction +0.0496 [+0.0000, +0.1385]; Wilcoxon p=0.5.
- **Oracle-Fixed-Filter**: final-regret reduction -0.0716 [-0.3144, +0.2003]; Wilcoxon p=0.625.

## 3. Source-region drift boundary

Positive regret reduction means the filter improves over Target-Only.

| Method | Drift | Mean reduction [95% bootstrap CI] | Mean trust | Activation |
|---|---:|---:|---:|---:|
| Adaptive-Filter | 0 | +0.1440 [+0.0000, +0.2879] | 0.207 | 100.0% |
| Adaptive-Filter | 0.5 | +0.1440 [+0.0000, +0.2879] | 0.158 | 100.0% |
| Adaptive-Filter | 1 | +0.0000 [+0.0000, +0.0000] | 0.111 | 100.0% |
| Adaptive-Filter | 2 | +0.0000 [+0.0000, +0.0000] | 0.185 | 100.0% |
| Fixed-Filter | 0 | +0.4132 [+0.3828, +0.4435] | 1.000 | 100.0% |
| Fixed-Filter | 0.5 | +0.0089 [-0.4258, +0.4435] | 1.000 | 100.0% |
| Fixed-Filter | 1 | -0.0218 [-0.4517, +0.4082] | 1.000 | 100.0% |
| Fixed-Filter | 2 | +0.1914 [+0.0000, +0.3828] | 1.000 | 100.0% |

### Data-derived boundary summary

- **Adaptive-Filter**: positive=none; negative=none; uncertain=[0.0, 0.5, 1.0, 2.0].
- **Fixed-Filter**: positive=[0.0]; negative=none; uncertain=[0.5, 1.0, 2.0].

## 4. Interpretation rule

The mechanism is supported only when matching-region filtering improves over both Target-Only and structure-matched random/wrong controls under paired tests, and when the sequential result preserves the same direction. Adaptive filtering is considered safer only if its negative-drift loss and wrong-source loss are smaller than fixed filtering.
