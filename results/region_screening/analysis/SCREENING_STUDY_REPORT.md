# Target-proposal / source-region-screening research report

This report is generated from experiment CSV files; conclusions are not hardcoded.

## 1. Shared-proposal mechanism study

All methods used the same target observations, target GP, raw pool, and target-proposed candidates.

| Method | Mean normalized regret | Top-10% hit rate | Mean retained fraction | Filter activation |
|---|---:|---:|---:|---:|
| Target-Only | 0.6742 | 7.8% | 1.000 | 0.0% |
| Matching-Fixed-Filter | 0.5578 | 15.6% | 0.250 | 100.0% |
| Matching-Adaptive-Filter | 0.6694 | 7.8% | 0.861 | 93.8% |
| Matching-Soft-Rerank | 0.5106 | 20.3% | 1.000 | 0.0% |
| Random-Adaptive-Filter | 0.6938 | 7.8% | 0.910 | 85.9% |
| Wrong-Adaptive-Filter | 0.6692 | 7.8% | 0.881 | 95.3% |
| Oracle-Fixed-Filter | 0.5879 | 10.9% | 0.754 | 32.8% |

### Paired comparisons against Target-Only

- **Matching-Fixed-Filter**: regret reduction +0.1164 [+0.0432, +0.1946]; Wilcoxon p=0.008286; paired win rate=53.1%.
- **Matching-Adaptive-Filter**: regret reduction +0.0048 [-0.0096, +0.0216]; Wilcoxon p=0.5751; paired win rate=9.4%.
- **Matching-Soft-Rerank**: regret reduction +0.1636 [+0.0833, +0.2522]; Wilcoxon p=0.0006628; paired win rate=62.5%.
- **Random-Adaptive-Filter**: regret reduction -0.0196 [-0.0464, +0.0000]; Wilcoxon p=0.1088; paired win rate=0.0%.
- **Wrong-Adaptive-Filter**: regret reduction +0.0051 [-0.0110, +0.0232]; Wilcoxon p=0.4631; paired win rate=6.2%.
- **Oracle-Fixed-Filter**: regret reduction +0.0863 [+0.0359, +0.1457]; Wilcoxon p=0.001847; paired win rate=25.0%.

## 2. Equal-budget sequential optimization

| Method | Final normalized regret | Total improvement | Filter activation | Mean trust |
|---|---:|---:|---:|---:|
| Target-Only | 0.5375 | 7.5723 | 0.0% | 0.000 |
| Matching-Fixed-Filter | 0.4690 | 8.2682 | 99.1% | 1.000 |
| Matching-Adaptive-Filter | 0.5010 | 7.7740 | 90.3% | 0.266 |
| Matching-Soft-Rerank | 0.3947 | 9.5384 | 0.0% | nan |
| Random-Adaptive-Filter | 0.5402 | 7.3075 | 71.9% | 0.130 |
| Wrong-Adaptive-Filter | 0.4797 | 7.8601 | 85.6% | 0.220 |
| Oracle-Fixed-Filter | 0.4814 | 8.2196 | 39.1% | 0.515 |

### Final-regret paired comparisons

- **Matching-Fixed-Filter**: final-regret reduction +0.0685 [+0.0114, +0.1281]; Wilcoxon p=0.04624.
- **Matching-Adaptive-Filter**: final-regret reduction +0.0365 [-0.0205, +0.0976]; Wilcoxon p=0.6169.
- **Matching-Soft-Rerank**: final-regret reduction +0.1428 [+0.0742, +0.2133]; Wilcoxon p=0.000615.
- **Random-Adaptive-Filter**: final-regret reduction -0.0027 [-0.0318, +0.0251]; Wilcoxon p=0.9375.
- **Wrong-Adaptive-Filter**: final-regret reduction +0.0578 [+0.0116, +0.1078]; Wilcoxon p=0.04767.
- **Oracle-Fixed-Filter**: final-regret reduction +0.0561 [+0.0178, +0.1020]; Wilcoxon p=0.02818.

## 3. Source-region drift boundary

Positive regret reduction means the filter improves over Target-Only.

| Method | Drift | Mean reduction [95% bootstrap CI] | Mean trust | Activation |
|---|---:|---:|---:|---:|
| Adaptive-Filter | 0 | -0.0017 [-0.0051, +0.0000] | 0.253 | 100.0% |
| Adaptive-Filter | 0.25 | +0.0000 [+0.0000, +0.0000] | 0.246 | 100.0% |
| Adaptive-Filter | 0.5 | -0.0017 [-0.0051, +0.0000] | 0.222 | 100.0% |
| Adaptive-Filter | 1 | +0.0160 [-0.0518, +0.0901] | 0.210 | 100.0% |
| Adaptive-Filter | 2 | -0.0000 [-0.0000, +0.0000] | 0.209 | 93.8% |
| Adaptive-Filter | 4 | +0.0608 [-0.0051, +0.1875] | 0.180 | 81.2% |
| Fixed-Filter | 0 | +0.0847 [-0.1084, +0.2848] | 1.000 | 100.0% |
| Fixed-Filter | 0.25 | +0.1074 [-0.0087, +0.2345] | 1.000 | 100.0% |
| Fixed-Filter | 0.5 | +0.1290 [-0.0306, +0.3013] | 1.000 | 100.0% |
| Fixed-Filter | 1 | +0.0176 [-0.1635, +0.1842] | 1.000 | 100.0% |
| Fixed-Filter | 2 | -0.0854 [-0.2150, +0.0263] | 1.000 | 100.0% |
| Fixed-Filter | 4 | -0.0460 [-0.1202, +0.0029] | 1.000 | 87.5% |

### Data-derived boundary summary

- **Adaptive-Filter**: positive=none; negative=none; uncertain=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0].
- **Fixed-Filter**: positive=none; negative=none; uncertain=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0].

## 4. Interpretation rule

The mechanism is supported only when matching-region filtering improves over both Target-Only and structure-matched random/wrong controls under paired tests, and when the sequential result preserves the same direction. Adaptive filtering is considered safer only if its negative-drift loss and wrong-source loss are smaller than fixed filtering.
