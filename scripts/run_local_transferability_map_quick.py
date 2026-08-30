"""Small exploratory map of source-local-model transferability.

This is deliberately descriptive, two-dimensional, and oracle-chart based.  It
uses a fitted source rank GP, controlled target deformations, and the transfer
models frozen in Pilot v1.  It is not a confirmatory study or an online-BO test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.local_surrogate_transfer import (  # noqa: E402
    LocalExpertResidualRegressor,
    LocalSurrogateTransferConfig,
)
from region_guided_reranking_study.local_surrogate_transfer_research import (  # noqa: E402
    evaluate_predictions,
    pairwise_cost_accuracy,
    rank_quality,
    sobol_chart_design,
)

Array = np.ndarray
METHODS = {
    "Target-Only": "target_only",
    "Fixed-Transfer": "fixed",
    "Calibrated-Transfer": "calibrated",
    "Gated-Transfer": "gated",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object.")
    return value


def rotation(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def local_cost(
    Z: Array,
    theta: float,
    scale: float = 1.0,
    weights: Tuple[float, float] = (1.0, 0.35),
    ripple: float = 0.07,
    frequencies: Tuple[float, float] = (3.0, 2.0),
) -> Array:
    points = np.asarray(Z, dtype=float)
    U = scale * (points @ rotation(theta).T)
    quadratic = weights[0] * U[:, 0] ** 2 + weights[1] * U[:, 1] ** 2
    oscillation = ripple * (
        1.0 - np.cos(frequencies[0] * np.pi * U[:, 0])
    ) + 0.5 * ripple * (
        1.0 - np.cos(frequencies[1] * np.pi * U[:, 1])
    )
    return quadratic + oscillation


def make_relation(
    relation: str,
    theta: float,
    independent_theta: float,
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    source = lambda Z: local_cost(Z, theta=theta)
    def independent(Z: Array) -> Array:
        points = np.asarray(Z, dtype=float)
        U = points @ rotation(independent_theta).T
        phase = 0.7 * independent_theta
        return (
            np.sin(3.3 * np.pi * U[:, 0] + phase)
            + 0.8 * np.cos(4.1 * np.pi * U[:, 1] - phase)
            + 0.35 * np.sin(2.0 * np.pi * (U[:, 0] + U[:, 1]))
        )
    if relation == "identity":
        return source, source
    if relation == "output_affine":
        return source, lambda Z: 4.0 + 2.5 * source(Z)
    if relation == "scale_0.7":
        return source, lambda Z: local_cost(Z, theta=theta, scale=0.7)
    if relation == "scale_1.5":
        return source, lambda Z: local_cost(Z, theta=theta, scale=1.5)
    if relation == "rotate_45":
        return source, lambda Z: local_cost(Z, theta=theta + np.pi / 4.0)
    if relation == "anisotropy_swap":
        return source, lambda Z: local_cost(
            Z,
            theta=theta,
            weights=(0.35, 1.0),
        )
    if relation == "roughness":
        return source, lambda Z: local_cost(
            Z,
            theta=theta,
            ripple=0.28,
            frequencies=(7.0, 5.0),
        )
    if relation == "reversal":
        return source, lambda Z: -source(Z)
    if relation == "independent_expert":
        return independent, source
    raise ValueError(f"Unknown relation: {relation}")


def fit_source_expert(
    X: Array,
    y: Array,
    length_scale: float,
    noise: float,
    seed: int,
) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(
            length_scale=np.full(2, length_scale),
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=True,
        random_state=seed,
    )
    model.fit(X, rank_quality(y))
    return model


def source_quality(model: GaussianProcessRegressor, X: Array) -> Array:
    return np.clip(np.asarray(model.predict(X), dtype=float).reshape(-1), 0.0, 1.0)


def transfer_config(config: Mapping, seed: int) -> LocalSurrogateTransferConfig:
    values = dict(config["transfer_model"])
    values["random_state"] = int(seed)
    return LocalSurrogateTransferConfig(**values)


def run_map(config: Mapping, output_dir: Path) -> pd.DataFrame:
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in config["seeds"]]
    relations = [str(value) for value in config["relations"]]
    n_source = int(config["source_train_samples"])
    n_context = int(config["target_context_samples"])
    n_test = int(config["target_test_samples"])
    top_fraction = float(config["top_fraction"])
    harm_margin = float(config["harm_margin_srmse"])
    expert_cfg = dict(config["source_expert"])

    rows: List[Dict] = []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20260830]))
        theta = float(rng.uniform(0.0, np.pi))
        independent_theta = float(theta + rng.uniform(0.35 * np.pi, 0.85 * np.pi))
        source_X = sobol_chart_design(2, n_source, seed=seed * 17 + 1)
        context_X = sobol_chart_design(2, n_context, seed=seed * 17 + 3)
        test_X = sobol_chart_design(2, n_test, seed=seed * 17 + 5)

        for relation_index, relation in enumerate(relations):
            expert_truth, target = make_relation(relation, theta, independent_theta)
            source_y = expert_truth(source_X)
            target_context_y = target(context_X)
            target_test_y = target(test_X)
            expert = fit_source_expert(
                source_X,
                source_y,
                length_scale=float(expert_cfg["length_scale"]),
                noise=float(expert_cfg["noise"]),
                seed=seed + relation_index * 101,
            )
            context_quality = source_quality(expert, context_X)
            test_quality = source_quality(expert, test_X)
            expert_test_cost = expert_truth(test_X)
            oracle_order_agreement = pairwise_cost_accuracy(
                expert_test_cost,
                target_test_y,
            )
            fitted_order_agreement = pairwise_cost_accuracy(
                1.0 - test_quality,
                target_test_y,
            )

            method_records: List[Dict] = []
            target_srmse = None
            for method, mode in METHODS.items():
                cfg = transfer_config(config, seed + relation_index * 1009)
                model = LocalExpertResidualRegressor(mode, cfg).fit(
                    context_X,
                    target_context_y,
                    None if mode == "target_only" else context_quality,
                )
                mean, std = model.predict(
                    test_X,
                    None if model.effective_mode_ == "target_only" else test_quality,
                    return_std=True,
                )
                metrics = evaluate_predictions(
                    target_test_y,
                    mean,
                    std,
                    top_fraction=top_fraction,
                )
                if method == "Target-Only":
                    target_srmse = metrics.standardized_rmse
                evidence = model.evidence_
                method_records.append(
                    {
                        "seed": seed,
                        "relation": relation,
                        "method": method,
                        **metrics.__dict__,
                        "oracle_order_agreement": oracle_order_agreement,
                        "fitted_expert_order_agreement": fitted_order_agreement,
                        "gate_accepted": bool(evidence.accepted) if evidence else False,
                        "gate_cv_gain": float(evidence.relative_rmse_gain) if evidence else np.nan,
                        "gate_context_pairwise": float(evidence.pairwise_accuracy) if evidence else np.nan,
                        "effective_mode": str(model.effective_mode_),
                    }
                )
            if target_srmse is None:
                raise RuntimeError("Target-Only result missing.")
            for record in method_records:
                record["srmse_delta_vs_target_only"] = (
                    record["standardized_rmse"] - target_srmse
                )
                record["negative_transfer"] = bool(
                    record["srmse_delta_vs_target_only"] > harm_margin
                )
                rows.append(record)

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "local_transferability_map_raw.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary_path = output_dir / "local_transferability_map_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_map(summary, output_dir / "local_transferability_map.png")
    write_report(summary, output_dir / "LOCAL_TRANSFERABILITY_MAP_CONCLUSION_CN.md")
    write_manifest(config, output_dir, [raw_path, summary_path, output_dir / "local_transferability_map.png", output_dir / "LOCAL_TRANSFERABILITY_MAP_CONCLUSION_CN.md"])
    return raw


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    base = raw[raw["method"] == "Target-Only"][
        ["seed", "relation", "standardized_rmse", "ndcg_at_top", "pairwise_accuracy"]
    ].rename(
        columns={
            "standardized_rmse": "base_srmse",
            "ndcg_at_top": "base_ndcg",
            "pairwise_accuracy": "base_pairwise",
        }
    )
    paired = raw.merge(base, on=["seed", "relation"], how="left")
    paired["srmse_advantage"] = paired["base_srmse"] - paired["standardized_rmse"]
    paired["ndcg_advantage"] = paired["ndcg_at_top"] - paired["base_ndcg"]
    paired["pairwise_advantage"] = paired["pairwise_accuracy"] - paired["base_pairwise"]
    return (
        paired.groupby(["relation", "method"], as_index=False)
        .agg(
            instances=("seed", "size"),
            oracle_order_agreement=("oracle_order_agreement", "mean"),
            fitted_expert_order_agreement=("fitted_expert_order_agreement", "mean"),
            mean_srmse=("standardized_rmse", "mean"),
            srmse_advantage=("srmse_advantage", "mean"),
            ndcg_advantage=("ndcg_advantage", "mean"),
            pairwise_advantage=("pairwise_advantage", "mean"),
            gate_acceptance=("gate_accepted", "mean"),
            negative_transfer_rate=("negative_transfer", "mean"),
        )
    )


def plot_map(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[summary["method"] == "Gated-Transfer"].copy()
    relations = selected["relation"].tolist()
    matrix = selected[
        [
            "fitted_expert_order_agreement",
            "srmse_advantage",
            "pairwise_advantage",
            "gate_acceptance",
            "negative_transfer_rate",
        ]
    ].to_numpy(dtype=float)
    matrix[:, 4] = 1.0 - matrix[:, 4]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=1.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(relations)), labels=relations)
    ax.set_xticks(
        range(5),
        labels=["expert-target\norder", "sRMSE\nadvantage", "pairwise\nadvantage", "gate\ncoverage", "1-harm\nrate"],
    )
    ax.set_title("Exploratory local-model transferability map (Gated Transfer)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    calibrated = summary[summary["method"] == "Calibrated-Transfer"].set_index("relation")
    gated = summary[summary["method"] == "Gated-Transfer"].set_index("relation")
    lines = [
        "# 局部模型可迁移性图谱：轻量探索结论",
        "",
        "这是 2D、8 seeds、8 条目标上下文的描述性探索，不是正式确认性实验，也不涉及在线 BO。",
        "",
        "| 局部关系 | 专家—目标排序一致率 | 校准 sRMSE 优势 | 校准 Pairwise 优势 | 门控接受率 | 门控负迁移率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for relation in calibrated.index:
        c = calibrated.loc[relation]
        g = gated.loc[relation]
        lines.append(
            f"| {relation} | {c['fitted_expert_order_agreement']:.3f} | "
            f"{c['srmse_advantage']:+.3f} | {c['pairwise_advantage']:+.3f} | "
            f"{g['gate_acceptance']:.3f} | {g['negative_transfer_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 简明判断",
            "",
            "- 当专家—目标排序一致率约为 0.93 以上时（identity、输出仿射、两档尺度变化），Pairwise 与 NDCG 均出现正增量。",
            "- 一致率降到约 0.77–0.79 时（旋转、粗糙度变化），收益很小且不同排序指标不完全一致；各向异性交换在一致率约 0.70 时已经有害。",
            "- 门控能完全拒绝独立专家与显式反转，但对中等程度结构失配不可靠：各向异性交换仍有 62.5% 接受率和 37.5% 负迁移率。",
            "",
            "## 决策",
            "",
            "局部代理迁移只在局部排序高度一致时值得使用。它不适合作为后续主线或立即接入在线 BO；最多保留为经过目标校准、并可精确回退 Target-Only 的辅助机制。若以后继续，门控应直接围绕目标侧局部排序证据设计，而不是使用任务级 matching/wrong 标签。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(config: Mapping, output_dir: Path, artifacts: List[Path]) -> None:
    payload = {
        "stage_id": config["stage_id"],
        "scope": "exploratory_2d_descriptive_not_online_bo",
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": {
            str(path.relative_to(REPO_ROOT)): file_hash(path) for path in artifacts
        },
    }
    (output_dir / "local_transferability_map_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local_transferability_map_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "local_transferability_map_quick",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    run_map(load_json(config_path.resolve()), args.output)
