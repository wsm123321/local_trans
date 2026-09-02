"""Strict semantic/reproducibility audit for Gate-0 quick artifacts.

The audit is intentionally tied to the frozen Gate-0 panel: 1080 result rows,
138240 candidate ledger rows, and 216 diagnostics rows.  It does not run the
quick runner and it treats ``relation_or_control`` as the collision-free
condition key for identity versus its label-permutation control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

# Import the runner's frozen generation primitives.  The audit deliberately
# does not execute the runner, but reconstruction must use the exact same
# implementation rather than a reimplemented approximation.
try:
    import run_oracle_local_model_transfer_quick as _runner
except Exception:  # pragma: no cover
    _runner = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
try:
    from region_guided_reranking_study.local_surrogate_transfer_research import evaluate_predictions
except Exception:  # pragma: no cover
    evaluate_predictions = None

from analyze_oracle_local_model_transfer_quick import (  # noqa: E402
    BOUNDARY_RELATIONS, CANONICAL_METHODS, DECISION_METRICS,
    NEGATIVE_RELATIONS, POSITIVE_RELATIONS, _contrasts, _normalise_ledger,
    _normalise_results, _summary, decide, locate, load_json,
)

EXPECTED_COUNTS = {"result_rows": 1080, "ledger_rows": 138240, "diagnostic_rows": 216}
EXPECTED_SEEDS = {11, 23, 37, 53, 71, 89, 107, 131}
EXPECTED_SHELLS = {"0.35", "0.7", "1.0"}
EXPECTED_CONDITIONS = set(POSITIVE_RELATIONS + NEGATIVE_RELATIONS + BOUNDARY_RELATIONS)
EXPECTED_METHODS = set(CANONICAL_METHODS)
EXPECTED_ARTIFACTS = {
    "results": "results.csv", "prediction_ledger": "prediction_ledger.csv",
    "source_expert_diagnostics": "source_expert_diagnostics.csv",
    "failures": "failures.csv", "config": "config.json",
    "reproducibility_inputs": "reproducibility_inputs.npz",
}
EXPECTED_DEPENDENCIES = {
    "runner", "oracle_core", "local_surrogate_transfer_research", "local_surrogate_transfer",
}
RESULT_KEY = ["dimension", "seed", "relation_or_control", "shell", "method"]
PANEL_KEY = ["dimension", "seed", "relation_or_control", "shell"]
DIAGNOSTIC_KEY = ["dimension", "seed", "relation_or_control", "shell"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for col in columns:
        if col in frame and pd.to_numeric(frame[col], errors="coerce").isna().any():
            bad.append(col)
    return bad


def _config_top_fraction(config: Mapping[str, Any]) -> float:
    if isinstance(config.get("pilot"), Mapping) and "top_fraction" in config["pilot"]:
        return float(config["pilot"]["top_fraction"])
    return float(config.get("top_fraction", 0.1))


def _panel_keys(frame: pd.DataFrame, include_method: bool = False) -> list[str]:
    keys = ["dimension", "dim", "seed", "relation_or_control", "control", "relation", "shell", "panel"]
    if include_method:
        keys.append("method")
    # Keep only one of dimension/dim and always retain relation_or_control.
    keys = [x for x in keys if x in frame.columns]
    if "relation_or_control" not in keys and "relation" in frame.columns:
        keys.remove("relation") if "relation" in keys else None
        keys.append("relation")
    return keys


def _metric_columns() -> list[str]:
    return ["standardized_rmse", "ndcg_at_top", "spearman", "pairwise_accuracy", "precision_at_top", "normalized_top1_regret", "mean_negative_log_likelihood", "interval_coverage_95"]


def _same_numeric(left: Iterable[Any], right: Iterable[Any], atol: float = 1e-9) -> bool:
    a = np.asarray(list(left), dtype=float); b = np.asarray(list(right), dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=1e-9, equal_nan=True))


def _hash_checks(input_dir: Path, paths: Mapping[str, Path], manifest: Mapping[str, Any], config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    expected: dict[str, str] = {}
    # New runner manifest uses logical artifact names.
    logical_names = {"results": "results.csv", "prediction_ledger": "prediction_ledger.csv", "source_expert_diagnostics": "source_expert_diagnostics.csv", "failures": "failures.csv"}
    for logical, digest in dict(manifest.get("artifacts", {})).items():
        if logical in logical_names:
            expected[str(input_dir / logical_names[logical])] = str(digest)
    for relative, digest in dict(manifest.get("artifact_sha256", {})).items():
        expected[str(relative)] = str(digest)
    bad = []
    for name, digest in expected.items():
        candidate = Path(name)
        if not candidate.is_absolute():
            local = input_dir / name
            candidate = local if local.exists() else ROOT / name.replace("\\", "/")
        if not candidate.exists() or file_sha256(candidate) != digest:
            bad.append(name)
    _check(checks, "artifact_hashes", not bad, bad or "all manifest artifacts match")
    if "config_sha256" in manifest:
        config_file = paths.get("config", input_dir / "config.json")
        actual = file_sha256(config_file) if config_file.exists() else "missing"
        _check(checks, "config_hash", actual == manifest["config_sha256"], f"expected={manifest['config_sha256']}, observed={actual}")
    # Validate optional runner/core hashes when a manifest declares a path and hash.
    for hash_key in ("runner_sha256", "core_sha256", "runner_hash", "core_hash"):
        if hash_key not in manifest:
            continue
        digest = str(manifest[hash_key]); path_key = hash_key.replace("_sha256", "_path").replace("_hash", "_path")
        declared_path = manifest.get(path_key) or manifest.get(hash_key.replace("_sha256", "_file"))
        if declared_path:
            p = Path(str(declared_path)); p = p if p.is_absolute() else ROOT / str(declared_path).replace("\\", "/")
            ok = p.exists() and file_sha256(p) == digest
            _check(checks, hash_key, ok, str(p))
        else:
            _check(checks, hash_key, False, "hash declared without corresponding path")


def _resolve_declared_path(input_dir: Path, value: Any) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    # Runner records repository-relative dependency paths, while artifact paths
    # are input-directory-relative.
    candidate = (ROOT / str(value).replace("\\", "/")).resolve()
    return candidate if candidate.exists() else (input_dir / p).resolve()


def _manifest_and_config_audit(input_dir: Path, paths: Mapping[str, Path], manifest: Mapping[str, Any], config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    """Validate provenance as a schema, not merely as an outer checksum."""
    expected_config = _runner.CANONICAL_CONFIG if _runner is not None else None
    config_path = paths.get("config")
    persisted = load_json(config_path) if config_path is not None and config_path.exists() else {}
    _check(checks, "persisted_config", config_path is not None and config_path.name == "config.json", "config.json is required")
    _check(checks, "exact_canonical_config", expected_config is not None and dict(persisted) == dict(expected_config), "config.json must equal runner.CANONICAL_CONFIG")
    _check(checks, "config_argument_matches_persisted", config_path is None or dict(config) == dict(persisted), "external config cannot override persisted config")
    _check(checks, "manifest_artifact_keys", set(manifest.get("artifacts", {})) == set(EXPECTED_ARTIFACTS), manifest.get("artifacts", {}))
    _check(checks, "manifest_dependency_keys", set(manifest.get("dependencies", {})) == EXPECTED_DEPENDENCIES, manifest.get("dependencies", {}))
    _check(checks, "manifest_dependency_sha256_keys", set(manifest.get("dependency_sha256", {})) == EXPECTED_DEPENDENCIES, manifest.get("dependency_sha256", {}))
    _check(checks, "manifest_config_exact", expected_config is not None and manifest.get("config") == expected_config, "manifest.config must equal runner.CANONICAL_CONFIG")
    _check(checks, "manifest_artifact_metadata_keys", set(manifest.get("artifact_metadata", {})) == {"config", "reproducibility_inputs"}, manifest.get("artifact_metadata", {}))
    if expected_config is not None:
        _check(checks, "manifest_scope", manifest.get("scope") == expected_config.get("scope"), manifest.get("scope"))
    for logical, filename in EXPECTED_ARTIFACTS.items():
        declared = manifest.get("artifacts", {}).get(logical)
        actual_path = input_dir / filename
        _check(checks, f"artifact_path_{logical}", actual_path.exists() and (config_path is None or (logical != "config" or config_path.resolve() == actual_path.resolve())), str(actual_path))
        _check(checks, f"artifact_hash_{logical}", actual_path.exists() and declared == file_sha256(actual_path), f"declared={declared}, path={actual_path}")
    if config_path and config_path.exists():
        _check(checks, "config_file_hash", manifest.get("config_sha256") == file_sha256(config_path), manifest.get("config_sha256"))
    required_versions = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import scipy, sklearn
        required_versions.update({"scipy": scipy.__version__, "sklearn": sklearn.__version__})
    except Exception:
        pass
    for name, version in required_versions.items():
        _check(checks, f"version_{name}", str(manifest.get(name, "")) == str(version), f"manifest={manifest.get(name)}, runtime={version}")
    deps = manifest.get("dependencies", {})
    for name in EXPECTED_DEPENDENCIES:
        item = deps.get(name, {})
        if isinstance(item, str): item = {"sha256": item}
        declared_path = item.get("path")
        p = _resolve_declared_path(input_dir, declared_path) if declared_path else Path("")
        ok = bool(declared_path) and p.exists() and item.get("sha256") == file_sha256(p)
        _check(checks, f"dependency_{name}", ok, f"path={p}, declared={item.get('sha256')}")
    # The manifest must describe every NPZ array losslessly: key set, shape,
    # dtype and _array_hash.  This catches stale or selectively edited metadata.
    npz_path = input_dir / EXPECTED_ARTIFACTS["reproducibility_inputs"]
    config_metadata = manifest.get("artifact_metadata", {}).get("config", {})
    _check(checks, "config_artifact_metadata", config_metadata == {"encoding": "utf-8", "indent": 2}, config_metadata)
    metadata = manifest.get("artifact_metadata", {}).get("reproducibility_inputs", {})
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                keys = sorted(z.files)
                shapes = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in keys}
                hash_keys = [k for k in keys if "target_" in k or "source_" in k or k in {"seed_lineage", "seed_values", "shell_values"}]
                hashes = {k: _runner._array_hash(z[k]) for k in hash_keys} if _runner is not None else {}
            _check(checks, "npz_metadata_keys", metadata.get("keys") == keys and metadata.get("array_count") == len(keys), f"expected={len(keys)}, declared={metadata.get('array_count')}")
            _check(checks, "npz_metadata_shapes", metadata.get("shapes") == shapes, "NPZ shape/dtype metadata")
            expected_hashes = hashes
            _check(checks, "npz_metadata_hashes", metadata.get("array_hashes") == expected_hashes, "NPZ array hashes")
        except Exception as exc:
            _check(checks, "npz_metadata", False, repr(exc))
    else:
        _check(checks, "npz_metadata", False, "reproducibility_inputs.npz missing")


def _semantic_hash_rules(results: pd.DataFrame, diagnostics: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    instance = [x for x in ["dimension", "dim", "seed"] if x in results]
    condition = "relation_or_control" if "relation_or_control" in results else "relation"
    context_hashes = [x for x in ["target_context_design_hash", "context_hash"] if x in results]
    test_hashes = [x for x in ["target_test_design_hash", "test_hash"] if x in results]
    truth_hashes = [x for x in ["target_test_truth_hash", "truth_hash"] if x in results]
    bad = 0
    if instance:
        for _, group in results.groupby(instance, dropna=False):
            # Context design is shared across all relation/control/shell/method.
            bad += sum(int(group[col].nunique(dropna=False) != 1) for col in context_hashes)
            if "shell" in group:
                for _, shell_group in group.groupby("shell", dropna=False):
                    bad += sum(int(shell_group[col].nunique(dropna=False) != 1) for col in test_hashes)
            if condition in group:
                truth_group_keys = [condition] + (["shell"] if "shell" in group else [])
                for _, condition_group in group.groupby(truth_group_keys, dropna=False):
                    bad += sum(int(condition_group[col].nunique(dropna=False) != 1) for col in truth_hashes)
    _check(checks, "shared_context_test_truth_hashes", bad == 0, f"violations={bad}")
    # Diagnostics must agree with result hashes on condition×shell panels.
    if not diagnostics.empty and instance and condition in diagnostics:
        result_hash = results.groupby(instance + [condition, "shell"], dropna=False)[[x for x in test_hashes + truth_hashes if x in results]].first().reset_index() if "shell" in results else pd.DataFrame()
        diag_hash = diagnostics.groupby([x for x in instance + [condition, "shell"] if x in diagnostics], dropna=False)[[x for x in test_hashes + truth_hashes if x in diagnostics]].first().reset_index() if "shell" in diagnostics else pd.DataFrame()
        diagnostic_bad = 0
        if len(result_hash) and len(diag_hash):
            merge_keys = [x for x in instance + [condition, "shell"] if x in result_hash and x in diag_hash]
            for column in merge_keys:
                result_hash[column] = result_hash[column].astype(str)
                diag_hash[column] = diag_hash[column].astype(str)
            merged = result_hash.merge(diag_hash, on=merge_keys, suffixes=("_r", "_d"))
            for col in test_hashes + truth_hashes:
                if f"{col}_r" in merged and f"{col}_d" in merged:
                    diagnostic_bad += int((merged[f"{col}_r"].astype(str) != merged[f"{col}_d"].astype(str)).sum())
        _check(checks, "diagnostic_hash_agreement", diagnostic_bad == 0, f"violations={diagnostic_bad}")


def _exact_panel_schema(results: pd.DataFrame, ledger: pd.DataFrame, diagnostics: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    result_required = set(RESULT_KEY + ["relation", "control", "panel", "model_seed"])
    # chart_point is a result-level provenance field in newer runner outputs;
    # accepting its absence would make cross-method identity unverifiable.
    result_required |= set(_metric_columns()) | {"effective_mode", "prior_names", "prior_coefficients", "source_data_hash", "source_query_transform", "source_context_query_hash", "source_test_query_hash", "target_context_design_hash", "target_test_design_hash", "target_test_truth_hash", "srmse_delta_vs_target_only", "negative_transfer"}
    ledger_required = set(PANEL_KEY + ["relation", "control", "panel", "candidate_index", "chart_point", "truth", "method", "model_seed", "predicted_mean", "predicted_std"])
    diagnostic_required = set(DIAGNOSTIC_KEY + ["relation", "control", "panel", "model_seed", "source_data_hash", "source_query_transform", "source_context_query_hash", "source_test_query_hash", "source_context_oob_count", "source_context_oob_rate", "source_test_oob_count", "source_test_oob_rate", "target_context_design_hash", "target_test_design_hash", "target_test_truth_hash", "source_permutation_hash", "source_value_pairwise_accuracy", "source_rank_target_agreement", "source_value_standardized_target_rmse"])
    _check(checks, "result_schema", result_required.issubset(results.columns), sorted(result_required - set(results.columns)))
    _check(checks, "ledger_schema", ledger_required.issubset(ledger.columns), sorted(ledger_required - set(ledger.columns)))
    _check(checks, "diagnostic_schema", diagnostic_required.issubset(diagnostics.columns), sorted(diagnostic_required - set(diagnostics.columns)))
    old = {"source_value_rmse", "source_rank_pairwise_accuracy"} & set(diagnostics.columns)
    _check(checks, "diagnostic_old_names_forbidden", not old, sorted(old))
    if len(results):
        expected = set(EXPECTED_CONDITIONS)
        mapping_bad = ((results.relation_or_control == "identity_label_permutation") & ((results.relation != "identity") | (results.control != "identity_label_permutation"))) | ((results.relation_or_control != "identity_label_permutation") & ((results.control != "none") | (results.relation != results.relation_or_control)))
        _check(checks, "relation_control_mapping", not bool(mapping_bad.any()) and set(results.relation_or_control) == expected, "identity control and relation labels")
    if len(diagnostics):
        mapping_bad = ((diagnostics.relation_or_control == "identity_label_permutation") & ((diagnostics.relation != "identity") | (diagnostics.control != "identity_label_permutation"))) | ((diagnostics.relation_or_control != "identity_label_permutation") & ((diagnostics.control != "none") | (diagnostics.relation != diagnostics.relation_or_control)))
        _check(checks, "diagnostic_relation_control_mapping", not bool(mapping_bad.any()), "diagnostics relation/control")
    # Exact frozen cartesian products and per-cell ledger cardinality.
    result_key = RESULT_KEY
    expected_result = len(EXPECTED_SEEDS) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SHELLS) * len(EXPECTED_METHODS)
    expected_result_keys = {(2, seed, condition, float(shell), method) for seed in EXPECTED_SEEDS for condition in EXPECTED_CONDITIONS for shell in (0.35, 0.7, 1.0) for method in results.method.unique() if str(method) in EXPECTED_METHODS}
    observed_result_keys = {(int(r.dimension), int(r.seed), str(r.relation_or_control), float(r.shell), str(r.method)) for r in results.itertuples()} if all(k in results for k in RESULT_KEY) else set()
    _check(checks, "result_exact_cartesian", len(results) == expected_result and observed_result_keys == {(2, seed, condition, float(shell), method) for seed in EXPECTED_SEEDS for condition in EXPECTED_CONDITIONS for shell in (0.35, 0.7, 1.0) for method in EXPECTED_METHODS} and not results.duplicated(result_key).any(), f"expected={expected_result}")
    if all(k in ledger for k in PANEL_KEY + ["method", "candidate_index"]):
        groups = ledger.groupby(PANEL_KEY + ["method"], dropna=False)
        expected_ledger_keys = {(2, seed, condition, float(shell), method) for seed in EXPECTED_SEEDS for condition in EXPECTED_CONDITIONS for shell in (0.35, 0.7, 1.0) for method in EXPECTED_METHODS}
        observed_ledger_keys = {(int(r.dimension), int(r.seed), str(r.relation_or_control), float(r.shell), str(r.method)) for r in ledger.itertuples()}
        bad = [key for key, part in groups if len(part) != 128 or set(pd.to_numeric(part.candidate_index, errors="coerce")) != set(range(128)) or not pd.api.types.is_integer_dtype(part.candidate_index)]
        _check(checks, "ledger_exact_candidates", not bad and observed_ledger_keys == expected_ledger_keys, f"bad_cells={bad[:3]}")
    if all(k in diagnostics for k in DIAGNOSTIC_KEY):
        expected_diag_keys = {(2, seed, condition, float(shell)) for seed in EXPECTED_SEEDS for condition in EXPECTED_CONDITIONS for shell in (0.35, 0.7, 1.0)}
        observed_diag_keys = {(int(r.dimension), int(r.seed), str(r.relation_or_control), float(r.shell)) for r in diagnostics.itertuples()}
        _check(checks, "diagnostic_exact_cartesian", len(diagnostics) == len(expected_diag_keys) and observed_diag_keys == expected_diag_keys and not diagnostics.duplicated(DIAGNOSTIC_KEY).any() and "method" not in diagnostics.columns, "216 cells without method")
    # Numeric domains are part of the schema, not only finiteness checks.
    for col in ("pairwise_accuracy", "ndcg_at_top", "precision_at_top", "interval_coverage_95", "source_value_pairwise_accuracy", "source_rank_target_agreement", "source_context_oob_rate", "source_test_oob_rate"):
        frame = diagnostics if col in diagnostics else results
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce")
            _check(checks, f"range_{col}", bool(values.notna().all() and ((values >= 0) & (values <= 1)).all()), "must lie in [0,1]")
    for col in ("source_context_oob_count", "source_test_oob_count"):
        if col in diagnostics:
            v = pd.to_numeric(diagnostics[col], errors="coerce")
            _check(checks, f"range_{col}", bool(v.notna().all() and (v >= 0).all() and (v.astype(int) == v).all()), "nonnegative integer")


def _npz_reconstruction(npz_path: Path, results: pd.DataFrame, ledger: pd.DataFrame, diagnostics: pd.DataFrame, config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    if _runner is None or not npz_path.exists():
        _check(checks, "npz_independent_reconstruction", False, "runner or NPZ unavailable"); return
    bad: list[str] = []
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            seeds = [int(x) for x in config["seeds"]]; shells = [float(x) for x in config["shells"]]; relations = [str(x) for x in config["relations"]]
            expected_keys = {"schema_version", "seed_values", "shell_values", "relation_names", "control_names", "seed_lineage", "source_dirs", "target_context_points", "target_test_points", "permutation", "theta"}
            for seed in seeds:
                expected_keys |= {f"seed_{seed}_{suffix}" for suffix in ("source_dirs", "context_dirs", "context_points", "test_dirs", "permutation", "theta", "theta_value", "independent_theta", "seed_lineage", "context_shell_index")}
                for relation in relations:
                    expected_keys |= {f"seed_{seed}_source_y_{relation}", f"seed_{seed}_source_context_query_{relation}", f"seed_{seed}_target_context_truth_{relation}"}
                    for shell in shells:
                        s = str(shell); expected_keys |= {f"seed_{seed}_{kind}_{relation}_{s}" for kind in ("target_test_points", "target_test_truth", "source_test_query")}
            _check(checks, "npz_exact_schema", set(z.files) == expected_keys, f"missing={sorted(expected_keys-set(z.files))[:3]}, extra={sorted(set(z.files)-expected_keys)[:3]}")
            _check(checks, "npz_frozen_arrays", np.array_equal(z["schema_version"], [1]) and np.array_equal(z["seed_values"], seeds) and np.allclose(z["shell_values"], shells) and list(z["relation_names"].astype(str)) == relations and list(z["control_names"].astype(str)) == ["identity_label_permutation"], "frozen schema arrays")
            for si, seed in enumerate(seeds):
                theta_seed = _runner._derive_seed(seed, "theta"); source_seed = _runner._derive_seed(seed, "source_design"); context_seed = _runner._derive_seed(seed, "target_context_design"); test_seed = _runner._derive_seed(seed, "target_test_design"); permutation_seed = _runner._derive_seed(seed, "source_permutation")
                theta_rng = np.random.default_rng(theta_seed); theta = float(theta_rng.uniform(0.0, np.pi)); independent_theta = float(theta + theta_rng.uniform(0.35*np.pi, 0.85*np.pi))
                source_dirs = _runner.sobol_chart_design(2, 128, seed=source_seed); context_dirs = _runner._unit_directions(12, context_seed); test_dirs = _runner._unit_directions(128, test_seed)
                context = np.vstack([context_dirs[i*4:(i+1)*4] * shell for i, shell in enumerate(shells)]); tests = {shell: test_dirs * shell for shell in shells}
                for shell in shells:
                    try:
                        _runner._assert_disjoint(context, tests[shell])
                    except Exception:
                        bad.append(f"{seed}:disjoint:{shell}")
                permutation = np.random.default_rng(permutation_seed).permutation(128).astype(np.int64)
                if not np.issubdtype(z[f"seed_{seed}_permutation"].dtype, np.integer) or not np.array_equal(np.sort(z[f"seed_{seed}_permutation"]), np.arange(128, dtype=np.int64)):
                    bad.append(f"{seed}:invalid_permutation")
                for name, expected in (("source_dirs", source_dirs), ("context_dirs", context_dirs), ("context_points", context), ("test_dirs", test_dirs), ("permutation", permutation), ("theta", np.array([theta, independent_theta])), ("theta_value", np.array([theta])), ("independent_theta", np.array([independent_theta]))):
                    if not np.array_equal(z[f"seed_{seed}_{name}"], expected): bad.append(f"{seed}:{name}")
                lineage = z[f"seed_{seed}_seed_lineage"].astype(np.int64).tolist(); expected_lineage = [seed, theta_seed, source_seed, _runner._derive_seed(seed,"source_expert"), context_seed, test_seed, permutation_seed] + [_runner._derive_seed(seed, f"model_{i}") for i in range(27)]
                if lineage != expected_lineage: bad.append(f"{seed}:lineage")
                if not np.array_equal(z["seed_lineage"][si], expected_lineage): bad.append(f"{seed}:stacked_lineage")
                for relation in relations:
                    source_truth, target_truth = _runner.make_relation(relation, theta, independent_theta); source_y = np.asarray(source_truth(source_dirs), dtype=float)
                    if not np.array_equal(z[f"seed_{seed}_source_y_{relation}"], source_y): bad.append(f"{seed}:{relation}:source_y")
                    if relation == "identity" and not np.array_equal(source_y[permutation], source_y[np.asarray(permutation, dtype=np.int64)]): bad.append(f"{seed}:identity:permutation")
                    transform = _runner.relation_transform(relation); source_context_query = transform(context)
                    if not np.array_equal(z[f"seed_{seed}_source_context_query_{relation}"], source_context_query): bad.append(f"{seed}:{relation}:context_query")
                    if not np.array_equal(z[f"seed_{seed}_target_context_truth_{relation}"], target_truth(context)): bad.append(f"{seed}:{relation}:context_truth")
                    controls = ["none", "identity_label_permutation"] if relation == "identity" else ["none"]
                    for control in controls:
                        condition = relation if control == "none" else control; train_y = source_y[permutation] if control != "none" else source_y
                        source_hash = _runner._array_hash(np.column_stack([source_dirs, train_y])); source_perm_hash = _runner._array_hash(permutation)
                        for shell_i, shell in enumerate(shells):
                            test = tests[shell]; test_truth = np.asarray(target_truth(test), dtype=float); source_test_query = transform(test); ctx_hash = _runner._array_hash(context); test_hash = _runner._array_hash(test); truth_hash = _runner._array_hash(test_truth); ctx_q_hash = _runner._array_hash(source_context_query); test_q_hash = _runner._array_hash(source_test_query)
                            key = (2, seed, condition, shell); rp = results[(results.dimension == 2) & (results.seed == seed) & (results.relation_or_control == condition) & (results.shell.astype(float) == shell)]
                            dp = diagnostics[(diagnostics.dimension == 2) & (diagnostics.seed == seed) & (diagnostics.relation_or_control == condition) & (diagnostics.shell.astype(float) == shell)]
                            for frame, label in ((rp, "result"), (dp, "diag")):
                                for col, value in (("source_data_hash",source_hash),("source_context_query_hash",ctx_q_hash),("source_test_query_hash",test_q_hash),("target_context_design_hash",ctx_hash),("target_test_design_hash",test_hash),("target_test_truth_hash",truth_hash)):
                                    if len(frame) and col in frame and set(frame[col].astype(str)) != {value}: bad.append(f"{key}:{label}:{col}")
                            if len(dp) and dp.source_permutation_hash.astype(str).iloc[0] != source_perm_hash: bad.append(f"{key}:permutation_hash")
                            if len(rp) and "model_seed" in rp:
                                expected_model = _runner._derive_seed(seed, f"model_{sum((2 if r == 'identity' and 'identity_label_permutation' in config['controls'] else 1) for r in relations[:relations.index(relation)]) + controls.index(control)}")
                                expected_model = _runner._derive_seed(seed, f"model_{(sum((2 if r == 'identity' and 'identity_label_permutation' in config['controls'] else 1) for r in relations[:relations.index(relation)]) + controls.index(control))*3 + shell_i}")
                                if set(pd.to_numeric(rp.model_seed, errors="coerce")) != {expected_model} or (len(dp) and set(pd.to_numeric(dp.model_seed, errors="coerce")) != {expected_model}): bad.append(f"{key}:model_seed")
                            if len(rp) and len(ledger):
                                lp = ledger[(ledger.dimension == 2)&(ledger.seed == seed)&(ledger.relation_or_control == condition)&(ledger.shell.astype(float) == shell)]
                                for _, row in rp.iterrows():
                                    m = lp[lp.method == row.method].sort_values("candidate_index")
                                    if len(m) != 128: bad.append(f"{key}:ledger_count"); continue
                                    points = np.asarray([json.loads(v) for v in m.chart_point], dtype=float); truth = pd.to_numeric(m.truth, errors="coerce").to_numpy(float)
                                    if not np.array_equal(points, test) or not np.allclose(truth, test_truth, atol=1e-12, rtol=0.0): bad.append(f"{key}:{row.method}:identity")
            # Exact lossless target truth hash is checked above from NPZ arrays.
    except Exception as exc:
        bad.append(repr(exc))
    _check(checks, "npz_independent_reconstruction", not bad, bad[:10])


def _diagnostic_recompute(npz_path: Path, diagnostics: pd.DataFrame, config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    if _runner is None or not npz_path.exists() or diagnostics.empty: return
    bad=[]
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            for seed in config["seeds"]:
                theta=float(z[f"seed_{seed}_theta_value"][0]); independent_theta=float(z[f"seed_{seed}_independent_theta"][0]); permutation=z[f"seed_{seed}_permutation"]
                source_dirs=z[f"seed_{seed}_source_dirs"]; context=z[f"seed_{seed}_context_points"]
                for relation in config["relations"]:
                    source_truth,target_truth=_runner.make_relation(relation,theta,independent_theta); source_y=np.asarray(source_truth(source_dirs)); train_options=[("identity_label_permutation",source_y[permutation])] if relation=="identity" else []
                    train_options.insert(0,("none",source_y))
                    for control,train_y in train_options:
                        cond=relation if control=="none" else control; transform=_runner.relation_transform(relation)
                        expert_cfg=dict(config["transfer_model"]); expert_cfg["gp_length_scale"]=float(config["source_expert"]["length_scale"]); expert_cfg["gp_noise"]=float(config["source_expert"]["noise"])
                        expert=_runner.fit_source_oracle_expert(source_dirs,train_y,_runner.OracleLocalModelTransferConfig(**expert_cfg),seed=_runner._derive_seed(seed,"source_expert"))
                        ctxq=transform(context); rankq,_=expert.predict_rank(ctxq); valueq,_=expert.predict(ctxq,feature="raw_value")
                        for shell in config["shells"]:
                            part=diagnostics[(diagnostics.seed==seed)&(diagnostics.relation_or_control==cond)&(diagnostics.shell.astype(float)==float(shell))]
                            testq=transform(z[f"seed_{seed}_target_test_points_{relation}_{shell}"]); testtruth=z[f"seed_{seed}_target_test_truth_{relation}_{shell}"]; rank,_=expert.predict_rank(testq); rawstd,_=expert.predict(testq,feature="raw_value"); raw=expert.raw_standardizer_.inverse_transform(rawstd) if expert.raw_standardizer_ is not None else rawstd
                            expected={"source_value_pairwise_accuracy":_runner.pairwise_cost_accuracy(raw,testtruth),"source_rank_target_agreement":_runner.pairwise_cost_accuracy(1.0-rank,testtruth),"source_value_standardized_target_rmse":float(np.sqrt(np.mean((rawstd-_runner._robust_standardize(testtruth))**2))),"source_context_oob_count":_runner._oob(ctxq)[0],"source_context_oob_rate":_runner._oob(ctxq)[1],"source_test_oob_count":_runner._oob(testq)[0],"source_test_oob_rate":_runner._oob(testq)[1]}
                            if len(part)!=1: bad.append(f"{seed}:{cond}:{shell}:count")
                            elif any(not np.isclose(float(part.iloc[0][k]),float(v),atol=1e-10,rtol=1e-10) for k,v in expected.items()): bad.append(f"{seed}:{cond}:{shell}:values")
    except Exception as exc: bad.append(repr(exc))
    _check(checks,"diagnostic_values_recomputed",not bad,bad[:10])


def _recompute_metrics(results: pd.DataFrame, ledger: pd.DataFrame, config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    if evaluate_predictions is None:
        _check(checks, "ledger_metrics_recomputed", False, "evaluate_predictions unavailable")
        return
    group = [x for x in ["dimension", "dim", "seed", "relation_or_control", "control", "relation", "shell", "panel", "method"] if x in ledger]
    # relation_or_control is mandatory for a collision-free audit; do not silently
    # fall back to relation when both identity conditions are present.
    if "relation_or_control" not in ledger:
        _check(checks, "collision_free_condition_key", False, "ledger lacks relation_or_control")
        return
    result_group = [x for x in group if x in results]
    errors: list[str] = []; compared = 0; max_diff = 0.0
    for key, part in ledger.groupby(group, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        mask = np.ones(len(results), dtype=bool)
        for col, value in zip(result_group, values):
            mask &= results[col].astype(str).eq(str(value)).to_numpy()
        candidate = results.loc[mask]
        if candidate.empty:
            errors.append(f"missing result row {key}"); continue
        row = candidate.iloc[0]
        truth = pd.to_numeric(part["truth"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(part["prediction"], errors="coerce").to_numpy(float)
        std = pd.to_numeric(part["predicted_std"], errors="coerce").to_numpy(float) if "predicted_std" in part and part["predicted_std"].notna().all() else None
        try:
            actual = evaluate_predictions(truth, prediction, std, top_fraction=_config_top_fraction(config)).__dict__
        except Exception as exc:
            errors.append(f"{key}: {exc}"); continue
        for metric in _metric_columns():
            if metric not in row or metric not in actual: continue
            observed = float(row[metric]); recomputed = float(actual[metric])
            if np.isfinite(observed) and np.isfinite(recomputed):
                compared += 1; max_diff = max(max_diff, abs(observed - recomputed))
                if not np.isclose(observed, recomputed, atol=1e-8, rtol=1e-8): errors.append(f"{key}:{metric}")
            elif not (np.isnan(observed) and np.isnan(recomputed)):
                errors.append(f"{key}:{metric}:nonfinite")
    _check(checks, "ledger_metrics_recomputed", not errors and compared > 0, f"compared={compared}, max_abs_diff={max_diff:.3e}, errors={errors[:5]}")


def _cross_method_identity(results: pd.DataFrame, ledger: pd.DataFrame, npz_path: Path, checks: list[dict[str, Any]]) -> None:
    """Every method must score the same lossless target panel."""
    bad = []
    if not all(k in ledger for k in PANEL_KEY + ["method", "candidate_index", "chart_point", "truth"]):
        _check(checks, "cross_method_candidate_identity", False, "ledger schema missing"); return
    for key, group in ledger.groupby(PANEL_KEY + ["candidate_index"], dropna=False):
        point_values = group.chart_point.astype(str).unique(); truths = pd.to_numeric(group.truth, errors="coerce").to_numpy(float)
        if len(point_values) != 1 or not np.allclose(truths, truths[0], atol=1e-12, rtol=0.0, equal_nan=False): bad.append(str(key))
    # model_seed is a panel-level lineage value and must agree in all three
    # persisted views, never merely be present in each file.
    if "model_seed" in results and "model_seed" in ledger:
        for key, rp in results.groupby(PANEL_KEY, dropna=False):
            seeds = set(pd.to_numeric(rp.model_seed, errors="coerce"))
            lp = ledger
            for col, value in zip(PANEL_KEY, key if isinstance(key, tuple) else (key,)): lp = lp[lp[col].astype(str) == str(value)]
            if seeds != set(pd.to_numeric(lp.model_seed, errors="coerce")): bad.append(f"model_seed:{key}")
    _check(checks, "cross_method_candidate_identity", not bad, bad[:5])


def _truth_consistency(results: pd.DataFrame, ledger: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    key = [x for x in ["dimension", "dim", "seed", "relation_or_control", "control", "relation", "shell", "panel"] if x in ledger]
    bad = 0
    for _, group in ledger.groupby(key, dropna=False):
        arrays = [np.sort(pd.to_numeric(part.truth, errors="coerce").to_numpy(float)) for _, part in group.groupby("method", dropna=False)]
        if arrays and any(not _same_numeric(arrays[0], arr, atol=1e-10) for arr in arrays[1:]): bad += 1
    _check(checks, "truth_consistency_across_methods", bad == 0, f"bad_panels={bad}")


def _fallback_check(results: pd.DataFrame, ledger: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    if "effective_mode" not in results:
        _check(checks, "target_only_fallback_exact", False, "results lacks effective_mode")
        return
    key = [x for x in ["dimension", "dim", "seed", "relation_or_control", "control", "relation", "shell", "panel"] if x in results]
    target_method = results.method == "target_only"
    fallback = results[(results.effective_mode.astype(str).str.lower() == "target_only") & ~target_method]
    target = results[target_method]
    ledger_key = [x for x in ["dimension", "dim", "seed", "relation_or_control", "control", "relation", "shell", "panel", "method"] if x in ledger]
    bad = 0
    for _, row in fallback.iterrows():
        base = target
        for col in key:
            base = base[base[col].astype(str).eq(str(row[col]))]
        if base.empty: bad += 1; continue
        target_row = base.iloc[0]
        for metric in _metric_columns():
            if metric in row and metric in target_row and pd.notna(row[metric]) and pd.notna(target_row[metric]) and not np.isclose(float(row[metric]), float(target_row[metric]), atol=1e-9, rtol=1e-9): bad += 1
        method_part = ledger
        for col in key:
            if col in method_part: method_part = method_part[method_part[col].astype(str).eq(str(row[col]))]
        got = method_part[method_part.method.astype(str).map(lambda x: str(x).lower()) == str(row.method).lower()]
        expected = method_part[method_part.method.astype(str).str.lower().eq("target_only")]
        if got.empty or expected.empty or not _same_numeric(got.sort_values("point_index").prediction, expected.sort_values("point_index").prediction, atol=1e-9): bad += 1
    _check(checks, "target_only_fallback_exact", bad == 0, f"mismatches={bad}")


def _permutation_semantics(results: pd.DataFrame, diagnostics: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    """Check that the identity control remains a true source-label permutation."""
    bad = 0
    if "relation_or_control" not in results or "control" not in results:
        _check(checks, "source_permutation_relation", False, "relation_or_control and control are required")
        return
    perm = results[results["relation_or_control"] == "identity_label_permutation"]
    if len(perm):
        bad += int((perm["relation"].astype(str) != "identity").sum())
        bad += int((perm["control"].astype(str) != "identity_label_permutation").sum())
    identity = results[results["relation_or_control"] == "identity"]
    # The target panel must be unchanged while source labels/data differ.
    join = [x for x in ["dimension", "seed", "shell", "panel"] if x in results]
    if len(perm) and len(identity) and join:
        source_col = "source_data_hash" if "source_data_hash" in results else None
        hash_cols = [x for x in ["target_context_design_hash", "target_test_design_hash", "target_test_truth_hash"] if x in results]
        left = identity.groupby(join, dropna=False).first().reset_index()
        right = perm.groupby(join, dropna=False).first().reset_index()
        merged = left.merge(right, on=join, suffixes=("_identity", "_permuted"))
        for col in hash_cols:
            if f"{col}_identity" in merged and f"{col}_permuted" in merged:
                bad += int((merged[f"{col}_identity"].astype(str) != merged[f"{col}_permuted"].astype(str)).sum())
        if source_col:
            bad += int((merged[f"{source_col}_identity"].astype(str) == merged[f"{source_col}_permuted"].astype(str)).sum())
    if len(diagnostics):
        diag = diagnostics[diagnostics["relation_or_control"] == "identity_label_permutation"] if "relation_or_control" in diagnostics else pd.DataFrame()
        if len(diag) and "source_permutation_hash" not in diag:
            bad += len(diag)
    _check(checks, "source_permutation_relation", bad == 0, f"bad_rows_or_panels={bad}")


def _coefficients_check(results: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    bad = 0
    for column in ["prior_coefficients", "coefficients"]:
        if column not in results: continue
        for value in results[column].dropna():
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                numbers = np.asarray(parsed, dtype=float).reshape(-1)
                bad += int(not np.all(np.isfinite(numbers)) or np.any(numbers < -1e-12))
            except (TypeError, ValueError, json.JSONDecodeError): bad += 1
    _check(checks, "coefficients_nonnegative", bad == 0, f"bad_rows={bad}")


def _analysis_recompute(input_dir: Path, results: pd.DataFrame, config: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    analysis = input_dir / "analysis"
    required = [analysis / x for x in ["summary.csv", "contrasts.csv", "relation_shell_headroom.png", "rank_value_dual_vs_geometry.png", "identity_observed_vs_permuted.png", "decision.json"]]
    _check(checks, "analysis_required_files", all(p.exists() for p in required), [str(p) for p in required if not p.exists()])
    if not all(p.exists() for p in required[:2]): return
    observed_summary = pd.read_csv(required[0]); observed_contrasts = pd.read_csv(required[1]); expected_summary = _summary(results); expected_contrasts = _contrasts(results)
    def same(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        if left.shape != right.shape or set(left.columns) != set(right.columns): return False
        cols = sorted(left.columns); a = left[cols].copy(); b = right[cols].copy(); sort = [x for x in ["seed", "relation_or_control", "control", "shell", "method", "baseline", "metric"] if x in cols]
        if sort: a = a.sort_values(sort, kind="stable").reset_index(drop=True); b = b.sort_values(sort, kind="stable").reset_index(drop=True)
        for col in cols:
            an = pd.to_numeric(a[col], errors="coerce"); bn = pd.to_numeric(b[col], errors="coerce")
            if an.notna().all() and bn.notna().all():
                if not np.allclose(an, bn, atol=1e-8, rtol=1e-8, equal_nan=True): return False
            elif not a[col].fillna("<NA>").astype(str).equals(b[col].fillna("<NA>").astype(str)): return False
        return True
    _check(checks, "analysis_summary_recomputed", same(observed_summary, expected_summary), f"observed={observed_summary.shape}, expected={expected_summary.shape}")
    _check(checks, "analysis_contrasts_recomputed", same(observed_contrasts, expected_contrasts), f"observed={observed_contrasts.shape}, expected={expected_contrasts.shape}")
    expected_label, expected_decision = decide(results, expected_contrasts, config)
    try: observed_decision = json.loads(required[5].read_text(encoding="utf-8"))
    except Exception: observed_decision = {}
    decision_ok = (
        observed_decision.get("label") == expected_label
        and observed_decision.get("selected_head") == expected_decision.get("selected_head")
        and observed_decision.get("rules") == expected_decision.get("rules")
        and observed_decision.get("head_evaluations") == expected_decision.get("head_evaluations")
        and observed_decision.get("head_to_head_evaluations") == expected_decision.get("head_to_head_evaluations")
    )
    _check(checks, "analysis_decision_recomputed", decision_ok, f"expected={expected_label}, observed={observed_decision.get('label')}")
    conclusion = input_dir / "ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md"
    text = conclusion.read_text(encoding="utf-8") if conclusion.exists() else ""
    required_conclusion_terms = [
        f"`{expected_label}`", "quick", "非正式", "seed mean", "CI",
        "seed win", "Value/Dual", "Rank", "negative", "Permutation",
        *POSITIVE_RELATIONS, *NEGATIVE_RELATIONS, *BOUNDARY_RELATIONS,
    ]
    conclusion_ok = all(term in text for term in required_conclusion_terms)
    # The conclusion is generated from the same decision payload: check every
    # reported numeric head-vs-Geometry and challenger-vs-Rank value occurs.
    for head, ev in expected_decision.get("head_evaluations", {}).items():
        for metric, detail in ev.get("positive_metric_detail", {}).items():
            conclusion_ok &= f"{float(detail['mean_delta']):+.5f}" in text
    for head, ev in expected_decision.get("head_to_head_evaluations", {}).items():
        for metric, detail in ev.get("metric_detail", {}).items():
            conclusion_ok &= f"{float(detail['mean_delta']):+.5f}" in text
    _check(checks, "conclusion_recomputed", conclusion_ok, f"label={expected_label}")
    if required[5].exists() and conclusion.exists():
        expected_hash = hashlib.sha256(conclusion.read_bytes()).hexdigest()
        _check(checks, "conclusion_hash", observed_decision.get("conclusion_sha256") == expected_hash, f"expected={expected_hash}, observed={observed_decision.get('conclusion_sha256')}" )
    # New analyzers may persist an explicit analysis manifest.  If present it
    # is checked completely; it is never used as a substitute for recomputing
    # source/results semantics (the manifest is not an adversarial trust anchor).
    analysis_manifest = analysis / "analysis_manifest.json"
    if analysis_manifest.exists():
        try:
            am = load_json(analysis_manifest)
            source_expected = {"results.csv": hashlib.sha256((input_dir / "results.csv").read_bytes()).hexdigest(), "prediction_ledger.csv": hashlib.sha256((input_dir / "prediction_ledger.csv").read_bytes()).hexdigest(), "source_expert_diagnostics.csv": hashlib.sha256((input_dir / "source_expert_diagnostics.csv").read_bytes()).hexdigest()}
            source = am.get("source_inputs", am.get("source_input_hashes", am.get("inputs", {}))); outputs = am.get("outputs", am.get("output_hashes", {}))
            def digest(entry: Any) -> Any:
                return entry.get("sha256") if isinstance(entry, Mapping) else entry
            source_ok = bool(source) and all(digest(entry) == file_sha256(_resolve_declared_path(input_dir, entry.get("path"))) if isinstance(entry, Mapping) and entry.get("path") and _resolve_declared_path(input_dir, entry.get("path")).exists() else False for entry in source.values())
            source_ok = source_ok and all(digest(source.get(k)) == v for k, v in source_expected.items())
            _check(checks, "analysis_manifest_source_hashes", source_ok, source)
            output_expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in required if p.exists()}
            output_ok = bool(outputs) and set(outputs) == set(output_expected) and all(digest(entry) == file_sha256(_resolve_declared_path(input_dir, entry.get("path"))) if isinstance(entry, Mapping) and entry.get("path") and _resolve_declared_path(input_dir, entry.get("path")).exists() else False for entry in outputs.values())
            output_ok = output_ok and all(digest(outputs.get(k)) == v for k, v in output_expected.items())
            _check(checks, "analysis_manifest_output_hashes", output_ok, outputs)
            analyzer_info = am.get("analyzer", {}); declared_analyzer = analyzer_info.get("sha256") if isinstance(analyzer_info, Mapping) else am.get("analyzer_sha256", am.get("analyzer_hash")); analyzer_path = analyzer_info.get("path") if isinstance(analyzer_info, Mapping) else am.get("analyzer_path", "scripts/analyze_oracle_local_model_transfer_quick.py")
            ap = _resolve_declared_path(input_dir, analyzer_path)
            _check(checks, "analysis_manifest_analyzer_hash", bool(declared_analyzer) and ap.exists() and declared_analyzer == file_sha256(ap), str(ap))
            conclusion_info = am.get("conclusion", {}); conclusion_declared = conclusion_info.get("sha256") if isinstance(conclusion_info, Mapping) else None
            conclusion_path = _resolve_declared_path(input_dir, conclusion_info.get("path")) if isinstance(conclusion_info, Mapping) and conclusion_info.get("path") else conclusion
            _check(checks, "analysis_manifest_conclusion_hash", bool(conclusion_declared) and conclusion_path == conclusion.resolve() and conclusion.exists() and conclusion_declared == file_sha256(conclusion), str(conclusion_path))
        except Exception as exc:
            _check(checks, "analysis_manifest", False, repr(exc))


def run_audit(input_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir).resolve(); paths = locate(input_dir)
    manifest = load_json(paths["manifest"]) if "manifest" in paths else {}
    config = load_json(Path(config_path).resolve()) if config_path and Path(config_path).exists() else (load_json(paths["config"]) if "config" in paths else manifest.get("config", {}))
    results = _normalise_results(pd.read_csv(paths["results"])); ledger = _normalise_ledger(pd.read_csv(paths["ledger"]))
    diagnostics = pd.read_csv(paths["diagnostics"]) if "diagnostics" in paths else pd.DataFrame()
    failures = pd.read_csv(paths["failures"]) if "failures" in paths else pd.DataFrame()
    checks: list[dict[str, Any]] = []
    _check(checks, "required_manifest", bool(manifest), paths.get("manifest", "missing"))
    _check(checks, "required_config", bool(config), paths.get("config", "embedded/missing"))
    _check(checks, "required_diagnostics", "diagnostics" in paths, paths.get("diagnostics", "missing"))
    _check(checks, "required_failures", "failures" in paths, paths.get("failures", "missing"))
    if manifest and config:
        _check(checks, "stage_identity", manifest.get("stage_id") == config.get("stage_id") == (_runner.CANONICAL_CONFIG.get("stage_id") if _runner is not None else config.get("stage_id")), f"manifest={manifest.get('stage_id')}, config={config.get('stage_id')}")
        _hash_checks(input_dir, paths, manifest, config, checks)
    _manifest_and_config_audit(input_dir, paths, manifest, config, checks)
    counts = {"result_rows": len(results), "ledger_rows": len(ledger), "diagnostic_rows": len(diagnostics)}
    _check(checks, "strict_rowcounts", counts == EXPECTED_COUNTS, f"expected={EXPECTED_COUNTS}, observed={counts}")
    if manifest.get("counts"):
        _check(checks, "manifest_rowcounts", all(int(manifest["counts"].get(k, v)) == v for k, v in counts.items()), manifest.get("counts"))
    _check(checks, "zero_failures", len(failures) == 0, f"failure_rows={len(failures)}")
    required_result_keys = [x for x in ["dimension", "seed", "relation_or_control", "control", "shell", "panel", "method"] if x in results]
    ledger_index_key = "candidate_index" if "candidate_index" in ledger.columns else "point_index"
    required_ledger_keys = [x for x in ["dimension", "seed", "relation_or_control", "control", "shell", "panel", ledger_index_key, "method"] if x in ledger]
    _check(checks, "collision_free_condition_key", "relation_or_control" in results and "relation_or_control" in ledger, "relation_or_control required")
    _check(checks, "result_keyset_unique", len(required_result_keys) == 7 and results.duplicated(required_result_keys).sum() == 0, f"key={required_result_keys}, duplicates={int(results.duplicated(required_result_keys).sum()) if required_result_keys else -1}")
    _check(checks, "ledger_keyset_unique", len(required_ledger_keys) == 8 and ledger.duplicated(required_ledger_keys).sum() == 0, f"key={required_ledger_keys}, duplicates={int(ledger.duplicated(required_ledger_keys).sum()) if required_ledger_keys else -1}")
    _check(checks, "panel_sets", set(results.seed.dropna().astype(int)) == EXPECTED_SEEDS and set(results.shell.astype(str)) == EXPECTED_SHELLS and set(results.relation_or_control) == EXPECTED_CONDITIONS and set(results.method) == EXPECTED_METHODS, "frozen seed/condition/shell/method panel")
    numeric_result = [x for x in _metric_columns() if x in results] + ["seed"]
    _check(checks, "finite_required", not _finite(results, numeric_result) and not _finite(ledger, ["truth", "prediction"]), "all required numeric values finite")
    # Diagnostics intentionally have no method column; avoid injecting the
    # analyzer's synthetic method="unknown" while normalizing them.
    if not diagnostics.empty:
        diagnostics = diagnostics.copy()
        diagnostics["shell"] = pd.to_numeric(diagnostics["shell"], errors="coerce") if "shell" in diagnostics else np.nan
        diagnostics["seed"] = pd.to_numeric(diagnostics["seed"], errors="coerce") if "seed" in diagnostics else np.nan
        diagnostics["dimension"] = pd.to_numeric(diagnostics["dimension"], errors="coerce") if "dimension" in diagnostics else np.nan
        if "relation_or_control" not in diagnostics and "relation" in diagnostics:
            diagnostics["relation_or_control"] = diagnostics["relation"]
    _exact_panel_schema(results, ledger, diagnostics, checks)
    _cross_method_identity(results, ledger, input_dir / "reproducibility_inputs.npz", checks)
    _npz_reconstruction(input_dir / "reproducibility_inputs.npz", results, ledger, diagnostics, config, checks)
    _diagnostic_recompute(input_dir / "reproducibility_inputs.npz", diagnostics, config, checks)
    _semantic_hash_rules(results, diagnostics, checks); _truth_consistency(results, ledger, checks); _recompute_metrics(results, ledger, config, checks); _fallback_check(results, ledger, checks); _coefficients_check(results, checks); _permutation_semantics(results, diagnostics, checks)
    _analysis_recompute(input_dir, results, config, checks)
    payload = {"ok": all(item["passed"] for item in checks), "stage_id": manifest.get("stage_id", config.get("stage_id")), "checks": checks, "counts": counts}
    (input_dir / "AUDIT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ("# Gate-0 Oracle Local-Model Transfer Quick Audit\n\n状态：**" + ("PASS" if payload["ok"] else "FAIL") + "**\n\n" + "审计限制：run_manifest.json 及 analysis_manifest.json 是被审计的声明性来源，不是对抗性信任锚；审计同时独立重算 CSV/NPZ 语义、哈希、指标、诊断和决策。\n\n" + "\n".join(f"- {'PASS' if c['passed'] else 'FAIL'} `{c['name']}`：{c['detail']}" for c in checks) + "\n")
    (input_dir / "FULL_RUN_AUDIT.md").write_text(report, encoding="utf-8")
    if not payload["ok"]: raise RuntimeError("Audit failed: " + ", ".join(c["name"] for c in checks if not c["passed"]))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=ROOT / "results" / "oracle_local_model_transfer_quick"); parser.add_argument("--config", type=Path); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); run_audit(args.input, args.config)
