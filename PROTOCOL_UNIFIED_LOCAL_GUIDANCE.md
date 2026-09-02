# Frozen Protocol: Unified Local-Guidance Study v1

## 1. Question and experimental unit

This study asks whether source local structure changes target candidate decisions and equal-budget sequential optimization when every source method sees the same extracted source library. The independent unit is `(problem, dimension, seed)`. Quick is a code-validation artifact; full is frozen before outcomes are inspected.

## 2. Frozen scale

- Quick: GMM/Ackley, 2D, 2 seeds, source sample count 80, raw pool 300, proposal panel 40, budget 5, bootstrap 500.
- Full: GMM/Rastrigin/Lunacek/Ackley × 2D/5D × 8 seeds, source sample count 160, raw pool 1500, proposal panel 100, budget 20, bootstrap 5000.
- Both use EI and `n_init=2d+2`. Full output identity is `results/unified_local_guidance_full`.

The four principal method names and meanings are exactly:

1. `Target-Only`: target acquisition only (`mode="target_only"`).
2. `Geometry-Only`: source geometry support only (`mode="geometry_only"`).
3. `Local-Rank-No-Reliability`: local rank surrogate without reliability weighting (`mode="local_rank_no_reliability"`).
4. `Local-Rank+Reliability`: local rank surrogate with the extractor's reliability weighting (`mode="local_rank_reliability"`).

The only required safety diagnostic is `Reversed-Local-Rank` (`mode="reversed_local_rank"`). A task-level wrong-source diagnostic may be added separately, but is not a principal method.

## 3. Source extraction and common guidance parameters

For each instance, source points and values are generated independently. One and only one `SourceLocalStructureLibrary` is extracted from those source observations and supplied unchanged to all three source methods and to the reversed diagnostic. No principal method uses a randomized source library. All source extractor parameters are explicit in the config `extraction` section, including elite/context sizes, GMM cap and regularization, model/CV settings, and reliability floors.

All guidance methods call `rank_local_structure_candidates` with the same `source_weight`, `target_nomination_ratio`, `source_nomination_ratio`, and `aggregation`; only `mode` changes. There is no target calibration, no adaptive gate, and no guidance parameter selected from target outcomes. `Reversed-Local-Rank` uses the same library and common parameters with only the reversed local-rank mode.

## 4. Leakage and randomization controls

Each instance starts with `SeedSequence([seed, dimension, stable_problem_code(problem)])`. Child streams are isolated for task construction, source design, target initialization, extractor state, mechanism panel, sequential steps, and observation noise. Source values are never used to fit a target GP. Target truth is revealed only for selected candidates or the frozen mechanism panel.

The mechanism phase constructs one target-only GP fitted on the common target initialization, one raw pool, one EI proposal panel, and one truth vector. Every mechanism method ranks that exact common panel. Candidate panels exclude target initialization, all source design points, and duplicates after 12-decimal rounding. Mechanism `true_rank` and `acquisition_rank` are 0-based with 0 denoting the best candidate. `raw_regret = selected_y - min(truth)` and `normalized_regret = raw_regret / max(q90(truth)-min(truth), 1e-12)`. `top10_hit` uses `truth <= quantile(truth, 0.10) + tolerance`; the panel's truth is explicitly an offline mechanism reveal.

The sequential phase gives every method its own target history and its own target-only GP at every step. A common preallocated step seed is used to make randomization reproducible, but histories differ, so proposal/candidate hashes are allowed to differ and are not claimed to match across methods. Every method receives the same budget and evaluates exactly one new target point per step.

## 5. Outputs

A non-empty output directory is rejected. The runner writes these required artifacts:

- `mechanism_results.csv` — one mechanism decision per instance/method, including the reversed diagnostic;
- `mechanism_candidate_panel.csv` — the shared mechanism proposal panel and truth;
- `sequential_summary.csv` — one final/AUC summary per sequential instance/method; AUC is computed over the complete trace at steps `0..budget`, and `trace_points=budget+1`;
- `sequential_traces.csv` — sequential best-value/regret traces including step 0;
- `source_structure_diagnostics.csv` — extracted library structure records;
- `failures.csv` — complete failed instance keys and exceptions;
- `run_manifest.json` — config/protocol/code hashes, Git HEAD and dirty status, Python/package identity, counts, and artifact hashes.

Additional compatibility artifacts are permitted. Failures are never silently dropped, and the runner never invokes analyzer or audit scripts. Existing results and the root README are not modified. Full parameters are not altered based on quick outcomes.

This protocol does not establish universal transfer, noisy-task robustness, general no-harm, or budget savings beyond the declared slices.
