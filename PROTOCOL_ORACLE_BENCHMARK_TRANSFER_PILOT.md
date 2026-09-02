# Frozen Protocol: Oracle Benchmark Transfer Pilot v1

## 1. Research Question and Scope

This pilot investigates whether oracle local value transfer provides statistically significant, non-trivial decision and surrogate prediction advantages over both geometric priors and oracle rank-only guidance across a diverse suite of 2D continuous optimization benchmarks.

Specifically, given an oracle source expert that captures true local source response characteristics, does transferring calibrated continuous values (`Oracle-Value+Residual`) outperform:
1. Spatial proximity alone (`Geometry-Prior+Residual`), and
2. Ordinal rank predictions alone (`Oracle-Rank+Residual`),
across decision metrics (Pairwise Accuracy, NDCG@top10%, Top-1 Regret Reduction) on held-out target local charts?

This is a **static held-out benchmark transfer evaluation**. It evaluates the theoretical upper bound and fundamental utility of local surrogate value transfer under known source-target basin correspondence before coupling with upstream region discovery or sequential acquisition loops.

---

## 2. Experimental Design & Factorial Space

### 2.1 Independent Experimental Unit
The independent statistical unit is:
```text
(problem, dimension, seed)
```
With 4 benchmark problems, 1 dimension ($D=2$), and 16 independent seeds, there are $4 \times 1 \times 16 = 64$ independent paired task instances.
Evaluations are evaluated across radial shells ($s \in \{0.35, 0.7, 1.0\}$). All metric measurements across shells within the same $(problem, dimension, seed, relation, context, method)$ instance are aggregated using equal-weighted arithmetic mean to yield exactly one observation per independent instance before method pairing. Shells are never treated as independent replicates ($N=64$).

### 2.2 Benchmark Problem Suite
Four standard continuous optimization benchmark landscapes with distinct topographic challenges:
1. **GMM**: Gaussian Mixture Model with multiple smooth local basins and varying basin curvatures.
2. **Rastrigin**: Highly multimodal landscape with a regular lattice of deceptive local minima surrounding the global basin.
3. **Lunacek**: Asymmetric double-basin problem designed to test attraction toward deceptive false global basins.
4. **Ackley**: Nearly flat outer landscape with steep exponential central bowl overlaid with high-frequency cosine oscillations.

### 2.3 Sample Budgets and Slices
- **Dimension**: $D = 2$.
- **Seeds (16 fixed seeds)**: `[42, 101, 2026, 777, 999, 1234, 5678, 8888, 31415, 27182, 13579, 24680, 11235, 44556, 77889, 90909]`.
- **Source Training Points**: $N_s = 128$ Sobol-sampled design points within the source local basin.
- **Target Context Sizes**: $N_c \in \{6, 12, 20\}$ nested Sobol prefix points.
  - **Primary Confirmatory Slice**: $N_c = 12$.
  - **Secondary Response Curves**: $N_c \in \{6, 20\}$.
- **Target Test Points**: $N_{\text{test}} = 512$ total held-out points across all radial shells evaluated within the target local chart.
- **Local Chart Radius**: $\text{chart\_radius\_fraction} = 0.04$, defining a physical radius $r = 0.04 \times \text{mean domain width}$.

### 2.4 Canonical Methods (5 Fixed Methods)
1. **`Target-Only`**: Local Gaussian Process regressor fitted strictly on the $N_c$ target context samples without any source transfer prior.
2. **`Geometry-Prior+Residual`**: Residual GP fitted on target context where the prior is an isotropic quadratic/distance penalty centered at the local basin center.
3. **`Oracle-Rank+Residual`**: Target GP residual model incorporating the source expert's ordinal rank quality prediction $h_s(x) \in [0, 1]$ via non-negative calibration.
4. **`Oracle-Value+Residual`**: Target GP residual model incorporating the source expert's standardized continuous value prediction $v_s(x)$ via bounded non-negative linear ridge calibration.
5. **`Oracle-Rank+Value+Residual`**: Dual residual GP incorporating both normalized rank quality and continuous value features simultaneously.

---

## 3. Oracle Construction, Leakage Controls & Isolation

### 3.1 Source Expert Construction
- The source expert is an exact Matérn-5/2 GP regressor fitted on the $N_s = 128$ source design points with lengthscale $\ell_s = 0.45$ and observation noise $\sigma^2_s = 10^{-4}$.
- **Rank Expert**: Outputs relative percentile rank quality $h_s(x) = \text{rank\_quality}(x) \in [0, 1]$ where higher represents superior objective value.
- **Value Expert**: Outputs robustly standardized continuous function values $v_s(x) = \frac{f_s(x) - \text{median}(y_s)}{\text{scale}(y_s)}$.

### 3.2 Target Calibration & Residual Model
- Target residual models fit $f_t(x) = \beta_0 + \beta_1 \phi_s(x) + \delta(x)$, where $\phi_s(x)$ is the source expert prediction.
- Bounded least-squares regression enforces $\beta_1 \ge 0$ with $L_2$ ridge regularization parameter $\lambda = 1.0$.
- The residual $\delta(x)$ is modeled with a target GP (Matérn-5/2, lengthscale $\ell_t = 0.6$, noise $\sigma^2_t = 10^{-4}$).

### 3.3 Leakage Prevention & Randomization Controls
- **RNG Architecture**: Deterministic SHA-256 stable derived pseudorandom streams generate independent, non-overlapping child streams for problem instantiation, source design, context sampling, test point generation, and model training.
- **Disjoint Panel Guarantee**: Target context ($N_c \le 20$) and held-out test points ($N_{\text{test}} = 512$ total across shells) are strictly disjoint, enforced by zero-overlap verification at 12-decimal precision.
- **Strict Blind Testing**: Test points and test objective values are completely hidden during all calibration and GP fitting stages; they are revealed exclusively to the offline evaluation metric suite.
- **Zero Cross-Contamination**: Source training samples are never pooled into the target context dataset.

---

## 4. Control Conditions & Negative Controls

To ensure falsifiability and detect false positive transfer artifacts, three transfer relations are evaluated:
1. **`matching`**: Source and target share corresponding landscape basins (True positive condition; used for confirmatory hypotheses).
2. **`reversed`** (Negative Control): Source expert output is inverted ($1 - h_s(x)$ for rank, $-v_s(x)$ for value). An effective transfer framework must not yield positive advantage over target-only/geometry under complete reversal.
3. **`label_permutation`** (Negative Control): Source function labels are permuted randomly while preserving spatial coordinates, destroying functional transferability while maintaining identical input geometry.

---

## 5. Confirmatory Hypotheses & Statistical Protocol

### 5.1 The 6 Primary Confirmatory Tests
All 6 primary tests are conducted on the confirmatory slice: `relation == "matching"` and `context_size == 12`.

| Hypothesis ID | Comparison | Baseline | Metric | Advantage Formulation |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Value vs Geometry | `Geometry-Prior+Residual` | Pairwise Accuracy | $\Delta = \text{Acc}_{\text{Value}} - \text{Acc}_{\text{Geometry}}$ |
| **H2** | Value vs Geometry | `Geometry-Prior+Residual` | NDCG@top10% | $\Delta = \text{NDCG}_{\text{Value}} - \text{NDCG}_{\text{Geometry}}$ |
| **H3** | Value vs Geometry | `Geometry-Prior+Residual` | Top-1 Regret Reduction | $\Delta = \text{Regret}_{\text{Geometry}} - \text{Regret}_{\text{Value}}$ |
| **H4** | Value vs Rank | `Oracle-Rank+Residual` | Pairwise Accuracy | $\Delta = \text{Acc}_{\text{Value}} - \text{Acc}_{\text{Rank}}$ |
| **H5** | Value vs Rank | `Oracle-Rank+Residual` | NDCG@top10% | $\Delta = \text{NDCG}_{\text{Value}} - \text{NDCG}_{\text{Rank}}$ |
| **H6** | Value vs Rank | `Oracle-Rank+Residual` | Top-1 Regret Reduction | $\Delta = \text{Regret}_{\text{Rank}} - \text{Regret}_{\text{Value}}$ |

### 5.2 Unified Positive Advantage Direction
In all 6 tests, advantage $\Delta$ is defined such that:
$$\Delta > 0 \iff \text{Oracle-Value is superior to the baseline}$$
- For higher-is-better metrics (Pairwise Accuracy, NDCG@top10%), $\Delta = \text{Value} - \text{Baseline}$.
- For lower-is-better metrics (Normalized Top-1 Regret), $\Delta = \text{Baseline} - \text{Value}$ (representing positive regret reduction).

### 5.3 Inferential Procedures & Support Decision Rule
- **Sample Unit**: Paired differences $\Delta_i$ across the $N = 64$ independent instances `(problem, dimension, seed)`.
- **Confidence Intervals**: 5000-resample nonparametric paired Bootstrap for mean advantage $\bar{\Delta}$ with 95% percentile confidence bounds $[\text{CI}_{0.025}, \text{CI}_{0.975}]$ (seed `20260902`).
- **Hypothesis Testing**: One-sided Wilcoxon signed-rank test with Pratt zero-handling testing $H_0: \Delta \le 0$ vs $H_1: \Delta > 0$ (`alternative="greater"`).
- **Multiple Testing Correction**: Holm-Bonferroni step-down procedure controlling the Family-Wise Error Rate (FWER) at $\alpha = 0.05$ across the family of 6 primary tests:
  $$\tilde{p}_{(k)} = \min\left(1, \max_{j \le k} ((6 - j + 1) p_{(j)})\right)$$
- **Support Criteria**: A hypothesis is declared **Supported** if and only if **BOTH**:
  $$\text{CI}_{0.025} > 0 \quad \text{AND} \quad \tilde{p} \le 0.05$$

---

## 6. Required Runner & Analyzer Artifacts

### 6.1 Runner Contract
The runner produces:
- `results.csv` — Matrix containing evaluation rows per shell `(problem, dimension, seed, relation, context_size, shell, method)` with standard metric columns (`pairwise_accuracy`, `ndcg_at_top`, `normalized_top1_regret`, `standardized_rmse`, `spearman`, `precision_at_top`).
- `source_expert_diagnostics.csv` — Diagnostics on source expert fit, calibration weights ($\beta_0, \beta_1$), and condition checks.
- `failures.csv` — Detailed failure records (empty if zero failures occurred).
- `config.json` — Persisted experiment configuration.

### 6.2 Analyzer Artifacts
The analysis pipeline generates:
1. `primary_tests.csv`: The 6 confirmatory hypothesis test outcomes with effect sizes, 95% CIs, raw Wilcoxon p-values, Holm-adjusted p-values, and Supported status flags.
2. `summary.csv`: Aggregated performance summary table across all conditions, context sizes, and methods (instance-level aggregation).
3. `problem_summary.csv`: Problem-level breakdown across GMM, Rastrigin, Lunacek, and Ackley (instance-level aggregation).
4. `report.md`: Markdown summary report with tables, safety checks, and formal statistical conclusions.
5. `figure1_primary_hypothesis_contrasts.png`: Forest plot of the 6 primary test effect sizes and 95% CIs, colored green if Supported and red otherwise.
6. `figure2_context_scaling_and_controls.png`: Multi-panel plot showing sample-efficiency curves across context sizes and negative control response based on instance means and SEMs.
7. `analysis_manifest.json`: Verification manifest with input/output SHA256 checksums and execution metadata.

---

## 7. Interpretation Boundaries & Scope Limitations

1. **Oracle Bound vs Practical Realizability**: These tests evaluate an oracle source expert (direct ground-truth function observations in the local chart). Practical surrogate transfer requires upstream regional discovery and coordinate alignment.
2. **Static Decision Quality vs Sequential Trajectories**: This protocol evaluates static held-out reranking and surrogate estimation on a fixed candidate pool. It does not measure active Bayesian optimization exploration-exploitation dynamics or cumulative regret over sequential iterations.
3. **Dimensionality**: Evaluated in $D=2$ local charts. Higher-dimensional scaling requires additional structural regularization.
4. **Local Basin Scope**: Conclusions apply strictly within the local chart ($r=0.04 \times \text{mean domain width}$) centered on relevant basins.
