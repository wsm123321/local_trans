"""Benchmark local chart adapter and primitives for oracle local-model transfer.

This module provides 2D oracle anchor extraction, exact coordinate mapping
(including exact rotation coordinate mapping for ShiftedRotatedRastrigin),
unit-sphere Sobol sampling and chart point generation, and adapter classes for
the four benchmark landscapes: GMM, Rastrigin, Lunacek, and Ackley.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .landscapes import (
    BaseLandscape,
    GaussianMixtureLandscape,
    LunacekBiRastrigin,
    ShiftedAckley,
    ShiftedRotatedRastrigin,
)
from .local_surrogate_transfer_research import sobol_chart_design

Array = np.ndarray

BENCHMARK_PROBLEMS = ("GMM", "Rastrigin", "Lunacek", "Ackley")
BENCHMARK_CONDITIONS = ("matching", "reversed", "label_permutation")
BENCHMARK_METHODS = (
    "Target-Only",
    "Geometry-Prior+Residual",
    "Oracle-Rank+Residual",
    "Oracle-Value+Residual",
    "Oracle-Rank+Value+Residual",
)
BENCHMARK_METHOD_MODES = {
    "Target-Only": "target_only",
    "Geometry-Prior+Residual": "geometry_prior",
    "Oracle-Rank+Residual": "oracle_rank",
    "Oracle-Value+Residual": "oracle_value",
    "Oracle-Rank+Value+Residual": "oracle_rank_value",
}


def derive_benchmark_seed(seed: int, stream: str) -> int:
    """Derive an explicit, stable 31-bit child seed from integer inputs."""
    payload = f"benchmark|{int(seed)}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def rastrigin_rotation_matrix(angle: float) -> Array:
    """Return 2D rotation matrix for given angle in radians."""
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.array([[c, -s], [s, c]], dtype=float)


def rastrigin_chart_transform(
    chart_points: Array,
    target_R: Array,
    source_R: Array,
) -> Array:
    """Exact target-to-source local chart rotation coordinate transform for Rastrigin.

    In Rastrigin, evaluated landscape value is:
    f(x) = 10*d + sum(Z^2 - 10*cos(2*pi*Z)), where Z = (x - shift) @ R.T.
    For target chart displacement u_tgt (where x_tgt = anchor_tgt + radius * u_tgt),
    Z_tgt = radius * (u_tgt @ target_R.T).
    To evaluate at identical Z in source, displacement u_src satisfies:
    u_src @ source_R.T = u_tgt @ target_R.T => u_src = u_tgt @ target_R.T @ source_R.
    """
    pts = np.asarray(chart_points, dtype=float)
    was_vector = (pts.ndim == 1)
    if was_vector:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 2:
        raise ValueError("Rastrigin 2D rotation requires 2-column chart points")
    transformed = pts @ target_R.T @ source_R
    return transformed.reshape(-1) if was_vector else transformed


def extract_oracle_anchor(landscape: BaseLandscape) -> Array:
    """Extract true global basin center as the 2D oracle anchor point."""
    basins = landscape.get_oracle_basins()
    if not basins:
        raise ValueError(f"Landscape {landscape.name} does not expose oracle basins")
    global_basins = [item for item in basins if bool(item.get("is_global", False))]
    selected = global_basins[0] if global_basins else max(
        basins,
        key=lambda item: float(item.get("weight", 0.0)),
    )
    return np.asarray(selected["center"], dtype=float).reshape(-1)


def compute_physical_chart_radius(
    landscape: BaseLandscape,
    chart_radius: Optional[float] = None,
    chart_radius_fraction: Optional[float] = None,
) -> float:
    """Compute physical chart radius from explicit radius or fraction of domain width."""
    if chart_radius is not None and float(chart_radius) > 0.0:
        return float(chart_radius)
    if chart_radius_fraction is not None and float(chart_radius_fraction) > 0.0:
        bounds = getattr(landscape, "bounds", None)
        if bounds is not None:
            bounds_arr = np.asarray(bounds, dtype=float)
            mean_width = float(np.mean(bounds_arr[:, 1] - bounds_arr[:, 0]))
            return float(chart_radius_fraction) * mean_width
        return float(chart_radius_fraction) * 10.0
    return 1.0


@dataclass(frozen=True)
class BenchmarkLandscapePair:
    """Benchmark target-source landscape pair with oracle anchors and local chart mapping."""

    problem: str
    target_landscape: BaseLandscape
    source_landscape: BaseLandscape
    target_anchor: Array
    source_anchor: Array
    chart_radius: float = 1.0
    target_rotation_matrix: Optional[Array] = None
    source_rotation_matrix: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.problem not in BENCHMARK_PROBLEMS:
            raise ValueError(f"Unsupported problem: {self.problem}. Must be one of {BENCHMARK_PROBLEMS}")
        t_anchor = np.asarray(self.target_anchor, dtype=float).reshape(-1)
        s_anchor = np.asarray(self.source_anchor, dtype=float).reshape(-1)
        if len(t_anchor) != len(s_anchor):
            raise ValueError("Target and source anchors must have matching dimension")
        if not np.all(np.isfinite(t_anchor)) or not np.all(np.isfinite(s_anchor)):
            raise ValueError("Anchors must contain finite values")
        if self.chart_radius <= 0.0:
            raise ValueError("chart_radius must be positive")
        object.__setattr__(self, "target_anchor", t_anchor)
        object.__setattr__(self, "source_anchor", s_anchor)

    @property
    def dim(self) -> int:
        return len(self.target_anchor)

    def chart_to_target_domain(self, chart_points: Array) -> Array:
        """Map local chart points to target domain physical coordinates."""
        pts = np.asarray(chart_points, dtype=float)
        was_vector = (pts.ndim == 1)
        if was_vector:
            pts = pts.reshape(1, -1)
        mapped = self.target_anchor[None, :] + self.chart_radius * pts
        return mapped.reshape(-1) if was_vector else mapped

    def target_to_source_chart(self, chart_points: Array) -> Array:
        """Map target local chart points to source local chart coordinates."""
        pts = np.asarray(chart_points, dtype=float)
        was_vector = (pts.ndim == 1)
        if was_vector:
            pts = pts.reshape(1, -1)
        if (
            self.problem == "Rastrigin"
            and self.target_rotation_matrix is not None
            and self.source_rotation_matrix is not None
        ):
            mapped = rastrigin_chart_transform(
                pts, self.target_rotation_matrix, self.source_rotation_matrix
            )
        else:
            mapped = pts.copy()
        return mapped.reshape(-1) if was_vector else mapped

    def chart_to_source_domain(self, chart_points: Array) -> Array:
        """Map target local chart points to source domain physical coordinates."""
        source_chart_pts = self.target_to_source_chart(chart_points)
        pts = np.asarray(source_chart_pts, dtype=float)
        was_vector = (pts.ndim == 1)
        if was_vector:
            pts = pts.reshape(1, -1)
        mapped = self.source_anchor[None, :] + self.chart_radius * pts
        return mapped.reshape(-1) if was_vector else mapped

    def evaluate_target(self, chart_points: Array) -> Array:
        """Evaluate target landscape on local chart points."""
        domain_pts = self.chart_to_target_domain(chart_points)
        return np.asarray(self.target_landscape(domain_pts), dtype=float).reshape(-1)

    def evaluate_source(self, chart_points: Array, condition: str = "matching") -> Array:
        """Evaluate source landscape for given chart points under specified condition."""
        domain_pts = self.chart_to_source_domain(chart_points)
        raw = np.asarray(self.source_landscape(domain_pts), dtype=float).reshape(-1)
        if condition == "reversed":
            return -raw
        return raw


def create_benchmark_pair(
    problem: str,
    seed: int,
    dim: int = 2,
    chart_radius: Optional[float] = None,
    chart_radius_fraction: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> BenchmarkLandscapePair:
    """Create reproducible benchmark target and source landscape pair with 2D oracle anchors."""
    if problem not in BENCHMARK_PROBLEMS:
        raise ValueError(f"Unknown problem: {problem}. Supported: {BENCHMARK_PROBLEMS}")
    if dim != 2:
        raise ValueError(f"Oracle benchmark transfer is configured for 2D, got dim={dim}")

    if rng is None:
        rng = np.random.default_rng(seed)

    if problem == "GMM":
        target = GaussianMixtureLandscape(dim=dim, rng=rng)
        perturbed_centers = [c + rng.normal(0, 0.15, size=dim) for c in target.centers]
        source = GaussianMixtureLandscape(
            dim=dim,
            centers=perturbed_centers,
            covs=target.covs,
            weights=target.weights,
            rng=rng,
        )
        target_anchor = extract_oracle_anchor(target)
        source_anchor = extract_oracle_anchor(source)
        radius = compute_physical_chart_radius(target, chart_radius, chart_radius_fraction)
        return BenchmarkLandscapePair(
            problem="GMM",
            target_landscape=target,
            source_landscape=source,
            target_anchor=target_anchor,
            source_anchor=source_anchor,
            chart_radius=radius,
        )

    if problem == "Rastrigin":
        target_shift = rng.uniform(-1.5, 1.5, size=dim)
        target_theta = float(rng.uniform(0.0, np.pi))
        target = ShiftedRotatedRastrigin(
            dim=dim,
            shift=target_shift,
            rotation_angle=target_theta,
            rng=rng,
        )
        source_shift = target_shift + rng.normal(0, 0.15, size=dim)
        source_theta = float(rng.uniform(0.0, np.pi))
        source = ShiftedRotatedRastrigin(
            dim=dim,
            shift=source_shift,
            rotation_angle=source_theta,
            rng=rng,
        )
        radius = compute_physical_chart_radius(target, chart_radius, chart_radius_fraction)
        return BenchmarkLandscapePair(
            problem="Rastrigin",
            target_landscape=target,
            source_landscape=source,
            target_anchor=target_shift,
            source_anchor=source_shift,
            chart_radius=radius,
            target_rotation_matrix=target.R,
            source_rotation_matrix=source.R,
        )

    if problem == "Lunacek":
        target_mu1 = rng.uniform(1.8, 2.5, size=dim)
        target = LunacekBiRastrigin(dim=dim, mu1=target_mu1, rng=rng)
        source_mu1 = target_mu1 + rng.normal(0, 0.15, size=dim)
        source = LunacekBiRastrigin(dim=dim, mu1=source_mu1, rng=rng)
        target_anchor = extract_oracle_anchor(target)
        source_anchor = extract_oracle_anchor(source)
        radius = compute_physical_chart_radius(target, chart_radius, chart_radius_fraction)
        return BenchmarkLandscapePair(
            problem="Lunacek",
            target_landscape=target,
            source_landscape=source,
            target_anchor=target_anchor,
            source_anchor=source_anchor,
            chart_radius=radius,
        )

    if problem == "Ackley":
        target_shift = rng.uniform(-1.2, 1.2, size=dim)
        target = ShiftedAckley(dim=dim, shift=target_shift, rng=rng)
        source_shift = target_shift + rng.normal(0, 0.15, size=dim)
        source = ShiftedAckley(dim=dim, shift=source_shift, rng=rng)
        target_anchor = extract_oracle_anchor(target)
        source_anchor = extract_oracle_anchor(source)
        radius = compute_physical_chart_radius(target, chart_radius, chart_radius_fraction)
        return BenchmarkLandscapePair(
            problem="Ackley",
            target_landscape=target,
            source_landscape=source,
            target_anchor=target_anchor,
            source_anchor=source_anchor,
            chart_radius=radius,
        )

    raise ValueError(f"Unhandled problem: {problem}")


def generate_unit_sphere_directions(n_points: int, seed: int, dim: int = 2) -> Array:
    """Generate reproducible unit sphere direction vectors.

    For 2D, uses 1D scrambled Sobol mapped to angles in [0, 2pi) to prevent
    the corner-concentration bias of hypercube radial projection.
    """
    if n_points < 1:
        raise ValueError("n_points must be positive")
    if dim == 2:
        unit = sobol_chart_design(1, n_points, seed=seed, lower=0.0, upper=1.0)
        theta = 2.0 * np.pi * unit[:, 0]
        return np.column_stack([np.cos(theta), np.sin(theta)])

    raw = sobol_chart_design(dim, n_points, seed=seed, lower=-1.0, upper=1.0)
    norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    return raw / norms


def generate_unit_ball_points(n_points: int, seed: int, dim: int = 2) -> Array:
    """Generate reproducible uniform unit ball/disk samples via Sobol.

    For 2D, uses 2D scrambled Sobol with polar radius r = sqrt(u_2) and angle theta = 2*pi*u_1
    to obtain an exact uniform distribution on the unit disk with no edge bias.
    """
    if n_points < 1:
        raise ValueError("n_points must be positive")
    if dim == 2:
        unit = sobol_chart_design(2, n_points, seed=seed, lower=0.0, upper=1.0)
        theta = 2.0 * np.pi * unit[:, 0]
        r = np.sqrt(unit[:, 1])
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    raw = sobol_chart_design(dim, n_points, seed=seed, lower=-1.0, upper=1.0)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    mask = (norms[:, 0] <= 1.0)
    if np.sum(mask) >= n_points:
        return raw[mask][:n_points]
    return raw[:n_points]


def partition_context_points(
    context_dirs: Array,
    shells: Sequence[float],
    n_context: int,
) -> Array:
    """Distribute n_context directions across shells using round-robin assignment.

    Ensures prefix stability: context[:6] matches C6, context[:12] matches C12, etc.
    """
    dirs = np.asarray(context_dirs, dtype=float)
    if len(dirs) < n_context:
        raise ValueError(f"context_dirs has {len(dirs)} points, need at least {n_context}")
    shells_list = [float(s) for s in shells]
    K = len(shells_list)
    if K == 0:
        raise ValueError("shells must be non-empty")
    if n_context < K:
        raise ValueError(f"n_context ({n_context}) must be >= number of shells ({K})")

    points = np.zeros((n_context, dirs.shape[1]), dtype=float)
    for j in range(n_context):
        shell_idx = j % K
        points[j] = dirs[j] * shells_list[shell_idx]
    return points


__all__ = [
    "BENCHMARK_PROBLEMS",
    "BENCHMARK_CONDITIONS",
    "BENCHMARK_METHODS",
    "BENCHMARK_METHOD_MODES",
    "BenchmarkLandscapePair",
    "derive_benchmark_seed",
    "rastrigin_rotation_matrix",
    "rastrigin_chart_transform",
    "extract_oracle_anchor",
    "compute_physical_chart_radius",
    "create_benchmark_pair",
    "generate_unit_sphere_directions",
    "generate_unit_ball_points",
    "partition_context_points",
]
