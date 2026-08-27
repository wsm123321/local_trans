"""Minimal runnable example of the local-region transfer optimizer."""

from __future__ import annotations

import os
import sys
import numpy as np

# Ensure src is in pythonpath
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from region_guided_reranking_study import (
    LocalRegionTransferConfig,
    LocalRegionTransferOptimizer,
)


def main() -> None:
    rng = np.random.default_rng(42)
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

    # Historical source task: its good local area is near (1.0, 1.0).
    source_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(120, 2))
    source_center = np.array([1.0, 1.0])
    source_y = np.sum((source_X - source_center) ** 2, axis=1)

    # Expensive target task: similar local area, with a moderate location shift.
    target_center = np.array([1.3, 0.8])

    def target_objective(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        quadratic = np.sum((X - target_center) ** 2, axis=1)
        local_ripple = 0.1 * np.sum(np.sin(3.0 * X) ** 2, axis=1)
        return quadratic + local_ripple

    init_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, 2))
    init_y = target_objective(init_X)

    optimizer = LocalRegionTransferOptimizer(
        bounds=bounds,
        config=LocalRegionTransferConfig(
            top_ratio=0.20,
            max_clusters=3,
            pool_size=500,
            source_weight=0.8,
            source_weight_decay=0.03,
            target_nomination_ratio=0.20,
            source_nomination_ratio=0.20,
            random_state=42,
        ),
    )

    region_library = optimizer.fit_source_regions(
        [(source_X, source_y)],
        task_ids=["historical_source"],
    )
    result = optimizer.optimize(
        objective=target_objective,
        init_X=init_X,
        init_y=init_y,
        budget=15,
    )

    print(f"Extracted source regions: {len(region_library.regions)}")
    print(f"Initial target best: {np.min(init_y):.6f}")
    print(f"Final target best:   {result.best_y:.6f}")
    print(f"Best target point:   {result.best_x}")


if __name__ == "__main__":
    main()
