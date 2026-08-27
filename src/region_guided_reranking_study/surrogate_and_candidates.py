"""
Target Surrogate Model and Candidate Pool Generation Module.
Implements:
1. Target Gaussian Process Surrogate with Acquisition Functions (EI, LCB, PI).
2. Pure independent Candidate Pool Generator with exclusion assertions for
   target initial samples and source historical points.
"""

from typing import Tuple, Optional, List
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm


class TargetGPSurrogate:
    """
    Gaussian Process surrogate model trained solely on target task observations D_t.
    """
    def __init__(self, dim: int, noise_level: float = 1e-4, random_state: int = 42):
        self.dim = dim
        self.noise_level = float(noise_level)
        self.random_state = int(random_state)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=2, 
            normalize_y=True,
            random_state=random_state
        )
        self.is_fitted = False
        self.y_min = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self.gp.fit(X, y)
        self.is_fitted = True
        self.y_min = float(np.min(y))

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.atleast_2d(X)
        if not self.is_fitted:
            raise RuntimeError("GP must be fitted before predict.")
        return self.gp.predict(X, return_std=return_std)

    def compute_acquisition(self, X: np.ndarray, acq_type: str = "ei", xi: float = 0.01, beta: float = 2.0) -> np.ndarray:
        """
        Compute acquisition score alpha_t(x). Higher is better.
        For minimization:
          - EI: Expected Improvement
          - LCB: Lower Confidence Bound (negated so higher is better)
          - PI: Probability of Improvement
        """
        mu, sigma = self.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        if acq_type.lower() == "ei":
            improvement = (self.y_min - mu - xi)
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(0.0, ei)
        elif acq_type.lower() == "lcb":
            return -(mu - beta * sigma)
        elif acq_type.lower() == "pi":
            z = (self.y_min - mu - xi) / sigma
            return norm.cdf(z)
        else:
            raise ValueError(f"Unknown acquisition type: {acq_type}")


class CandidatePoolGenerator:
    """
    Generates candidate pool C_t independently from source/target random streams.
    Guarantees no overlap with evaluated target points or source samples.
    """
    def __init__(self, bounds: np.ndarray, pool_size: int = 1000, 
                 ratio_acq: float = 0.4, ratio_global: float = 0.4, ratio_diverse: float = 0.2,
                 rng: Optional[np.random.Generator] = None):
        self.bounds = np.array(bounds, dtype=float)
        self.dim = bounds.shape[0]
        self.pool_size = pool_size
        self.ratio_acq = ratio_acq
        self.ratio_global = ratio_global
        self.ratio_diverse = ratio_diverse
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def generate(self, surrogate: Optional[TargetGPSurrogate] = None, 
                 current_X: Optional[np.ndarray] = None,
                 excluded_datasets: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Generate candidate pool C_t of exact shape (pool_size, dim).
        Enforces strict exclusion of current_X and excluded_datasets.
        """
        n_global = int(self.pool_size * self.ratio_global)
        n_diverse = int(self.pool_size * self.ratio_diverse)
        n_acq = self.pool_size - n_global - n_diverse
        
        candidates = []
        
        # 1. Global uniform
        c_global = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_global, self.dim))
        candidates.append(c_global)
        
        # 2. Diverse perturbations
        if current_X is not None and len(current_X) > 0:
            idx_sampled = self.rng.choice(len(current_X), size=n_diverse, replace=True)
            scales = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
            noise = self.rng.normal(0, scales, size=(n_diverse, self.dim))
            c_diverse = current_X[idx_sampled] + noise
            c_diverse = np.clip(c_diverse, self.bounds[:, 0], self.bounds[:, 1])
        else:
            c_diverse = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_diverse, self.dim))
        candidates.append(c_diverse)
        
        # 3. Acquisition-focused screening
        if surrogate is not None and surrogate.is_fitted:
            screening_batch = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_acq * 6, self.dim))
            acq_scores = surrogate.compute_acquisition(screening_batch, acq_type="ei")
            top_acq_idx = np.argsort(acq_scores)[-n_acq:]
            c_acq = screening_batch[top_acq_idx]
        else:
            c_acq = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_acq, self.dim))
        candidates.append(c_acq)
        
        pool = np.vstack(candidates)
        pool = np.clip(pool, self.bounds[:, 0], self.bounds[:, 1])
        
        # Assert exclusion: filter out points that are too close to current_X or excluded_datasets
        all_excluded = []
        if current_X is not None and len(current_X) > 0:
            all_excluded.append(current_X)
        if excluded_datasets is not None:
            for ds in excluded_datasets:
                if len(ds) > 0:
                    all_excluded.append(ds)
                    
        if all_excluded:
            stacked_excluded = np.vstack(all_excluded)
            # Remove exact or extremely close points (dist < 1e-5)
            filtered_pts = []
            for pt in pool:
                dists = np.linalg.norm(stacked_excluded - pt[np.newaxis, :], axis=1)
                if np.min(dists) > 1e-5:
                    filtered_pts.append(pt)
                else:
                    # Replace with a fresh random point
                    new_pt = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
                    filtered_pts.append(new_pt)
            pool = np.array(filtered_pts)
            
        assert pool.shape == (self.pool_size, self.dim), f"Candidate pool shape mismatch: {pool.shape}"
        return pool
