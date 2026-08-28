# ARISE-BO: decision-conditional local transferability report

The report is generated from CSV files. Positive regret reduction means improvement over target-only BO.

## 1. Region-identification quality

| Scenario | Policy | AUROC | AUPRC | Brier | Gain Spearman | Trusted precision | Rejected precision |
|---|---|---:|---:|---:|---:|---:|---:|
| matching | arise | 0.527 | 0.309 | 0.260 | 0.022 | nan | nan |
| matching | fixed | 0.544 | 0.440 | 0.275 | 0.046 | 0.700 | 1.000 |
| matching | global_adaptive | 0.470 | 0.237 | 0.208 | -0.032 | 0.368 | 1.000 |
| matching | posterior | 0.542 | 0.315 | 0.273 | 0.099 | 0.000 | nan |
| matching | target_only | 0.542 | 0.315 | 0.273 | 0.099 | 0.000 | nan |
| mixed | arise | 0.542 | 0.334 | 0.260 | 0.005 | 0.545 | nan |
| mixed | fixed | 0.578 | 0.315 | 0.258 | 0.091 | 0.333 | nan |
| mixed | global_adaptive | 0.441 | 0.243 | 0.214 | -0.073 | 1.000 | nan |
| mixed | posterior | 0.615 | 0.313 | 0.257 | 0.156 | nan | nan |
| mixed | target_only | 0.615 | 0.313 | 0.257 | 0.156 | nan | nan |
| wrong | arise | 0.618 | 0.570 | 0.251 | 0.186 | 1.000 | nan |
| wrong | fixed | 0.557 | 0.307 | 0.282 | -0.000 | 0.250 | nan |
| wrong | global_adaptive | 0.510 | 0.363 | 0.239 | -0.000 | 0.625 | 0.500 |
| wrong | posterior | 0.621 | 0.349 | 0.269 | 0.098 | nan | nan |
| wrong | target_only | 0.621 | 0.349 | 0.269 | 0.098 | nan | nan |

## 2. Equal-budget optimization

| Scenario | Policy | Final normalized regret | Regret AUC | Mean improvement | Global steps | Probe steps | Exploit steps |
|---|---|---:|---:|---:|---:|---:|---:|
| matching | arise | 0.8007 | 0.9423 | 0.7552 | 0.00 | 2.75 | 0.00 |
| matching | fixed | 0.4235 | 0.6886 | 6.4486 | 0.00 | 0.00 | 0.00 |
| matching | global_adaptive | 0.6220 | 0.7633 | 0.7392 | 7.00 | 0.00 | 0.00 |
| matching | posterior | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |
| matching | target_only | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |
| mixed | arise | 0.6851 | 0.8737 | 0.8561 | 0.00 | 3.00 | 0.75 |
| mixed | fixed | 0.5716 | 0.8440 | 5.8094 | 0.00 | 0.00 | 0.00 |
| mixed | global_adaptive | 0.6951 | 0.8699 | 3.5900 | 8.00 | 0.00 | 0.00 |
| mixed | posterior | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |
| mixed | target_only | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |
| wrong | arise | 0.8106 | 0.9177 | 2.5143 | 0.00 | 3.00 | 0.25 |
| wrong | fixed | 0.4398 | 0.7681 | 5.7040 | 0.00 | 0.00 | 0.00 |
| wrong | global_adaptive | 0.6492 | 0.8123 | 1.3650 | 7.75 | 0.00 | 0.00 |
| wrong | posterior | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |
| wrong | target_only | 0.7416 | 0.8768 | 3.0780 | 0.00 | 0.00 | 0.00 |

## 3. Paired comparisons against target-only

### matching

- **fixed**: reduction +0.3181 [+0.1402, +0.4735], Wilcoxon p=0.125, win=100.0%, loss=0.0%.
- **global_adaptive**: reduction +0.1196 [-0.1887, +0.4279], Wilcoxon p=0.625, win=50.0%, loss=50.0%.
- **posterior**: reduction +0.0000 [+0.0000, +0.0000], Wilcoxon p=1, win=0.0%, loss=0.0%.
- **arise**: reduction -0.0591 [-0.2398, +0.0836], Wilcoxon p=0.75, win=25.0%, loss=50.0%.
### mixed

- **fixed**: reduction +0.1700 [-0.0621, +0.4021], Wilcoxon p=0.625, win=50.0%, loss=50.0%.
- **global_adaptive**: reduction +0.0465 [-0.1634, +0.3033], Wilcoxon p=1, win=25.0%, loss=50.0%.
- **posterior**: reduction +0.0000 [+0.0000, +0.0000], Wilcoxon p=1, win=0.0%, loss=0.0%.
- **arise**: reduction +0.0565 [-0.2398, +0.4304], Wilcoxon p=1, win=25.0%, loss=50.0%.
### wrong

- **fixed**: reduction +0.3018 [-0.0163, +0.5889], Wilcoxon p=0.25, win=75.0%, loss=25.0%.
- **global_adaptive**: reduction +0.0924 [-0.1252, +0.3100], Wilcoxon p=0.5, win=50.0%, loss=25.0%.
- **posterior**: reduction +0.0000 [+0.0000, +0.0000], Wilcoxon p=1, win=0.0%, loss=0.0%.
- **arise**: reduction -0.0691 [-0.3048, +0.1343], Wilcoxon p=0.75, win=25.0%, loss=50.0%.

## 4. Interpretation

ARISE is supported only when its region posterior is calibrated, trusted-region precision is high, and the full method improves over both target-only and fixed guidance in the mixed/wrong-source scenarios. A useful region is defined by positive counterfactual decision gain, not merely geometric or global task similarity.