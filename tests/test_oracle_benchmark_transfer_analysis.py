"""Unit and integration tests for Oracle Benchmark Transfer Pilot analysis."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_oracle_benchmark_transfer_pilot as analyzer  # noqa: E402


# -----------------------------------------------------------------------------
# Unit Tests for Statistical Procedures
# -----------------------------------------------------------------------------

def test_holm_bonferroni_correction_known_values() -> None:
    """Test that Holm step-down correction produces exact theoretical bounds."""
    p_raw = [0.005, 0.010, 0.012, 0.030, 0.040, 0.050]
    p_adj, _ = analyzer.holm_bonferroni_correction(p_raw)
    
    assert len(p_adj) == 6
    assert np.isclose(p_adj[0], 0.030, atol=1e-5)
    assert np.isclose(p_adj[1], 0.050, atol=1e-5)
    assert np.isclose(p_adj[2], 0.050, atol=1e-5)
    assert np.isclose(p_adj[3], 0.090, atol=1e-5)
    assert np.isclose(p_adj[4], 0.090, atol=1e-5)
    assert np.isclose(p_adj[5], 0.090, atol=1e-5)
    assert np.all(np.diff(p_adj) >= -1e-12)


def test_holm_bonferroni_order_invariance() -> None:
    """Test that Holm adjustment preserves original index positions."""
    p_raw = [0.04, 0.001, 0.02, 0.50]
    p_adj, order = analyzer.holm_bonferroni_correction(p_raw)
    
    assert p_adj[1] < p_adj[2] < p_adj[0] < p_adj[3]
    assert np.isclose(p_adj[1], 0.004, atol=1e-5)


def test_advantage_direction_unified_positive() -> None:
    """Test that advantage direction is consistently positive when Value is superior."""
    rows = []
    for p in ["GMM", "Rastrigin", "Lunacek", "Ackley"]:
        for s in [1, 2, 3, 4]:
            rows.extend([
                {
                    "problem": p, "dimension": 2, "seed": s, "relation": "matching", "context_size": 12, "shell": 1.0,
                    "method": "Oracle-Value+Residual", "method_key": "value",
                    "pairwise_accuracy": 0.85, "ndcg_at_top": 0.92, "normalized_top1_regret": 0.03,
                },
                {
                    "problem": p, "dimension": 2, "seed": s, "relation": "matching", "context_size": 12, "shell": 1.0,
                    "method": "Geometry-Prior+Residual", "method_key": "geometry",
                    "pairwise_accuracy": 0.65, "ndcg_at_top": 0.70, "normalized_top1_regret": 0.18,
                },
                {
                    "problem": p, "dimension": 2, "seed": s, "relation": "matching", "context_size": 12, "shell": 1.0,
                    "method": "Oracle-Rank+Residual", "method_key": "rank",
                    "pairwise_accuracy": 0.75, "ndcg_at_top": 0.80, "normalized_top1_regret": 0.10,
                },
            ])
    df = pd.DataFrame(rows)
    primary_df = analyzer.evaluate_primary_hypotheses(df, relation="matching", context_size=12, n_bootstrap=500)
    
    assert len(primary_df) == 6
    for _, row in primary_df.iterrows():
        assert row["mean_advantage"] > 0, f"Hypothesis {row['hypothesis_id']} advantage should be positive"
        assert row["ci_lower_95"] > 0, f"Hypothesis {row['hypothesis_id']} CI lower should be positive"
        assert row["p_raw_wilcoxon"] < 0.01, f"Hypothesis {row['hypothesis_id']} p-value should be significant"
        assert row["significant_fwer"] is True, f"Hypothesis {row['hypothesis_id']} should be significant"
        assert row["supported"] is True, f"Hypothesis {row['hypothesis_id']} should be supported"


def test_safe_wilcoxon_pratt_edge_cases() -> None:
    """Test safe Wilcoxon signed-rank test under zero differences and boundary conditions."""
    zeros = [0.0] * 20
    assert analyzer.safe_wilcoxon_greater(zeros) == 1.0

    positives = [0.05, 0.10, 0.15, 0.08, 0.12, 0.09, 0.14, 0.11]
    p_pos = analyzer.safe_wilcoxon_greater(positives)
    assert p_pos < 0.01

    negatives = [-0.10, -0.05, -0.08, -0.12, -0.09]
    p_neg = analyzer.safe_wilcoxon_greater(negatives)
    assert p_neg > 0.90


def test_bootstrap_mean_ci_properties() -> None:
    """Test bootstrap confidence interval estimation properties."""
    arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    mean_val, ci_lower, ci_upper = analyzer.bootstrap_mean_ci(arr, n_bootstrap=1000, confidence=0.95, seed=42)
    
    assert np.isclose(mean_val, 5.5)
    assert ci_lower < mean_val < ci_upper
    assert 1.0 <= ci_lower
    assert ci_upper <= 10.0


def test_multi_shell_aggregation_preserves_independent_instances() -> None:
    """Test that multiple shell rows are aggregated by mean per instance and do not inflate n_instances."""
    rows = []
    for prob, seed in [("GMM", 42), ("Ackley", 101)]:
        for shell, shell_factor in [(0.35, 0.0), (0.70, 0.02), (1.00, 0.04)]:
            rows.append({
                "problem": prob, "dimension": 2, "seed": seed, "relation": "matching", "context_size": 12, "shell": shell,
                "method": "Oracle-Value+Residual", "method_key": "value",
                "pairwise_accuracy": 0.80 + shell_factor, "ndcg_at_top": 0.85, "normalized_top1_regret": 0.05,
            })
            rows.append({
                "problem": prob, "dimension": 2, "seed": seed, "relation": "matching", "context_size": 12, "shell": shell,
                "method": "Geometry-Prior+Residual", "method_key": "geometry",
                "pairwise_accuracy": 0.60 + shell_factor, "ndcg_at_top": 0.65, "normalized_top1_regret": 0.15,
            })
            rows.append({
                "problem": prob, "dimension": 2, "seed": seed, "relation": "matching", "context_size": 12, "shell": shell,
                "method": "Oracle-Rank+Residual", "method_key": "rank",
                "pairwise_accuracy": 0.70 + shell_factor, "ndcg_at_top": 0.75, "normalized_top1_regret": 0.10,
            })
            
    df = pd.DataFrame(rows)
    assert len(df) == 18
    
    primary_df = analyzer.evaluate_primary_hypotheses(df, relation="matching", context_size=12, n_bootstrap=500)
    assert len(primary_df) == 6
    assert all(primary_df["n_instances"] == 2)
    
    h1 = primary_df[primary_df["hypothesis_id"] == "H1_Matching_Pairwise_Value_vs_Geometry"].iloc[0]
    assert np.isclose(h1["mean_value"], 0.82, atol=1e-5)
    assert np.isclose(h1["mean_baseline"], 0.62, atol=1e-5)
    assert np.isclose(h1["mean_advantage"], 0.20, atol=1e-5)


def test_supported_decision_requires_both_ci_and_p_value() -> None:
    """Test that a hypothesis is NOT supported if CI lower <= 0, even if p-value is nominally significant."""
    records = [
        {
            "hypothesis_id": "H1_Test",
            "comparison": "Value vs Geometry",
            "baseline": "Geometry",
            "baseline_key": "geometry",
            "metric": "pairwise_accuracy",
            "metric_display": "Pairwise",
            "relation": "matching",
            "context_size": 12,
            "n_instances": 10,
            "mean_value": 0.70,
            "mean_baseline": 0.69,
            "mean_advantage": 0.01,
            "ci_lower_95": -0.02,
            "ci_upper_95": 0.04,
            "p_raw_wilcoxon": 0.01,
            "alpha": 0.05,
            "higher_is_better": True,
            "description": "Test",
        }
    ]
    raw_p = [0.01]
    adj_p, _ = analyzer.holm_bonferroni_correction(raw_p)
    
    records[0]["p_adjusted_holm"] = float(adj_p[0])
    records[0]["significant_fwer"] = bool(adj_p[0] <= 0.05)
    records[0]["supported"] = bool((adj_p[0] <= 0.05) and (records[0]["ci_lower_95"] > 0))
    
    assert records[0]["significant_fwer"] is True
    assert records[0]["supported"] is False


# -----------------------------------------------------------------------------
# End-to-End Synthetic Pipeline Integration Test
# -----------------------------------------------------------------------------

def _create_synthetic_runner_output(target_dir: Path) -> None:
    """Create a fully compliant synthetic runner directory with shells, diagnostics, and safety flags."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    problems = ["GMM", "Rastrigin", "Lunacek", "Ackley"]
    dims = [2]
    seeds = [42, 101]
    relations = ["matching", "reversed", "label_permutation"]
    context_sizes = [6, 12, 20]
    shells = [0.35, 0.70, 1.00]
    methods = [
        "Target-Only",
        "Geometry-Prior+Residual",
        "Oracle-Rank+Residual",
        "Oracle-Value+Residual",
        "Oracle-Rank+Value+Residual",
    ]
    
    rng = np.random.default_rng(2026)
    rows = []
    
    for prob in problems:
        for d in dims:
            for s in seeds:
                for rel in relations:
                    for ctx in context_sizes:
                        for sh in shells:
                            for meth in methods:
                                effective_mode = "calibrated"
                                neg_transfer = False
                                if rel == "matching":
                                    if meth == "Oracle-Value+Residual":
                                        acc = 0.88 + rng.uniform(0.01, 0.05)
                                        ndcg = 0.91 + rng.uniform(0.01, 0.04)
                                        regret = 0.02 + rng.uniform(0.00, 0.01)
                                    elif meth == "Oracle-Rank+Value+Residual":
                                        acc = 0.89 + rng.uniform(0.01, 0.05)
                                        ndcg = 0.92 + rng.uniform(0.01, 0.04)
                                        regret = 0.02 + rng.uniform(0.00, 0.01)
                                    elif meth == "Oracle-Rank+Residual":
                                        acc = 0.78 + rng.uniform(0.01, 0.04)
                                        ndcg = 0.81 + rng.uniform(0.01, 0.04)
                                        regret = 0.08 + rng.uniform(0.01, 0.03)
                                    elif meth == "Geometry-Prior+Residual":
                                        acc = 0.68 + rng.uniform(0.01, 0.04)
                                        ndcg = 0.72 + rng.uniform(0.01, 0.04)
                                        regret = 0.14 + rng.uniform(0.01, 0.03)
                                    else:
                                        acc = 0.60 + 0.005 * ctx + rng.uniform(0.0, 0.03)
                                        ndcg = 0.65 + 0.005 * ctx + rng.uniform(0.0, 0.03)
                                        regret = 0.20 - 0.003 * ctx + rng.uniform(0.0, 0.02)
                                        effective_mode = "target_only"
                                elif rel == "reversed":
                                    if "Oracle" in meth:
                                        effective_mode = "target_only"
                                        neg_transfer = False
                                    acc = 0.60 + rng.uniform(0.0, 0.05)
                                    ndcg = 0.65 + rng.uniform(0.0, 0.05)
                                    regret = 0.20 + rng.uniform(0.0, 0.05)
                                else:  # label_permutation
                                    if "Oracle" in meth:
                                        effective_mode = "target_only" if rng.uniform() < 0.375 else "calibrated"
                                        neg_transfer = rng.uniform() < 0.4375
                                    acc = 0.50 + rng.uniform(-0.03, 0.03)
                                    ndcg = 0.55 + rng.uniform(-0.03, 0.03)
                                    regret = 0.30 + rng.uniform(0.0, 0.05)

                                rows.append({
                                    "problem": prob,
                                    "dimension": d,
                                    "seed": s,
                                    "relation": rel,
                                    "context_size": ctx,
                                    "shell": sh,
                                    "method": meth,
                                    "pairwise_accuracy": acc,
                                    "ndcg_at_top": ndcg,
                                    "normalized_top1_regret": regret,
                                    "standardized_rmse": 0.1 + rng.uniform(0.01, 0.05),
                                    "spearman": 0.7 + rng.uniform(0.01, 0.05),
                                    "precision_at_top": 0.8 + rng.uniform(0.01, 0.05),
                                    "effective_mode": effective_mode,
                                    "negative_transfer": neg_transfer,
                                    "srmse_delta_vs_target_only": -0.05 if not neg_transfer else 0.05,
                                })
                            
    results_df = pd.DataFrame(rows)
    results_df.to_csv(target_dir / "results.csv", index=False)
    
    diag_rows = [
        {"problem": prob, "seed": s, "source_expert_rmse": 0.01, "beta_1": 1.0}
        for prob in problems for s in seeds
    ]
    pd.DataFrame(diag_rows).to_csv(target_dir / "source_expert_diagnostics.csv", index=False)
    
    pd.DataFrame(columns=["problem", "dimension", "seed", "method", "error"]).to_csv(
        target_dir / "failures.csv", index=False
    )
    
    config_data = {
        "stage_id": "oracle-benchmark-transfer-pilot-v1",
        "problems": problems,
        "dimension": 2,
        "seeds": seeds,
        "source_samples": 128,
        "context_sizes": context_sizes,
        "primary_context_size": 12,
        "target_test_samples": 512,
        "relations": relations,
        "methods": methods,
        "analysis": {
            "bootstrap_samples": 500,
            "alpha": 0.05,
            "bootstrap_seed": 20260902,
        },
    }
    (target_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")


def test_end_to_end_analysis_pipeline(tmp_path: Path) -> None:
    """Test end-to-end analyzer execution on synthetic runner outputs with multi-shell data."""
    runner_dir = tmp_path / "runner_out"
    _create_synthetic_runner_output(runner_dir)
    
    out_dir = tmp_path / "analysis_out"
    result = analyzer.run_analysis(
        input_dir=runner_dir,
        output_dir=out_dir,
        n_bootstrap=500,
        alpha=0.05,
        seed=20260902,
    )
    
    assert (out_dir / "primary_tests.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "problem_summary.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "oracle_benchmark_transfer_pilot_report.md").exists()
    assert (out_dir / "figure1_primary_hypothesis_contrasts.png").exists()
    assert (out_dir / "figure2_context_scaling_and_controls.png").exists()
    assert (out_dir / "analysis_manifest.json").exists()

    # Verify report contains problem heterogeneity and control sections
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Benchmark Problem-Level Heterogeneity" in report_text
    assert "Negative Control & Safety Analysis" in report_text
    assert "Non-negative slope constraints alone cannot replace empirical cross-validation gating" in report_text

    primary_df = pd.read_csv(out_dir / "primary_tests.csv")
    assert len(primary_df) == 6
    assert all(primary_df["n_instances"] == 8)
    assert "supported" in primary_df.columns

    summary_df = pd.read_csv(out_dir / "summary.csv")
    assert summary_df["pairwise_accuracy_count"].max() == 8
    
    assert (out_dir / "figure1_primary_hypothesis_contrasts.png").stat().st_size > 1000
    assert (out_dir / "figure2_context_scaling_and_controls.png").stat().st_size > 1000

    manifest = json.loads((out_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_id"] == "oracle-benchmark-transfer-pilot-v1"
    assert "analyzer" in manifest and "sha256" in manifest["analyzer"]
    assert "results" in manifest["inputs"]
    assert "diagnostics" in manifest["inputs"]
    assert "config" in manifest["inputs"]
    assert "primary_tests.csv" in manifest["outputs"]


def test_analyzer_handles_column_aliases_and_missing_failures(tmp_path: Path) -> None:
    """Test that analyzer gracefully handles column aliases and missing failure logs."""
    runner_dir = tmp_path / "runner_alias"
    runner_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for meth in ["Target-Only", "Geometry-Prior+Residual", "Oracle-Rank+Residual", "Oracle-Value+Residual", "Oracle-Rank+Value+Residual"]:
        rows.append({
            "task": "GMM",
            "dim": 2,
            "seed": 42,
            "condition": "matching",
            "n_context": 12,
            "policy": meth,
            "pairwise": 0.85 if "Value" in meth else 0.70,
            "ndcg": 0.90 if "Value" in meth else 0.75,
            "regret": 0.05 if "Value" in meth else 0.15,
            "srmse": 0.10,
            "spearman_rho": 0.80,
            "top_precision": 0.85,
        })
    pd.DataFrame(rows).to_csv(runner_dir / "results.csv", index=False)
    
    out_dir = tmp_path / "analysis_alias_out"
    res = analyzer.run_analysis(input_dir=runner_dir, output_dir=out_dir, n_bootstrap=200)
    
    assert (out_dir / "primary_tests.csv").exists()
    assert (out_dir / "report.md").exists()
    primary_df = pd.read_csv(out_dir / "primary_tests.csv")
    assert len(primary_df) == 6
    assert all(primary_df["n_instances"] == 1)
