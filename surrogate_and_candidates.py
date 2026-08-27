"""
Target Surrogate Model and Shared Candidate Pool Generation.
Implements:
1. Target Gaussian Process Surrogate with Standard Acquisition Functions (EI, UCB, PI).
2. Broad Shared Candidate Pool Generator: C_t = C_acq U C_global U C_diverse.
"""

from typing import Tuple, Optional
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
        self.noise_level = noise_level
        self.random_state = random_state
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5, 
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
        Compute acquisition score alpha_t(x).
        Higher acquisition score means more preferred.
        For minimization:
          - EI: E[max(0, y_best - f(x) - xi)]
          - LCB / UCB: - (mu - beta * sigma)
          - PI: P(f(x) <= y_best - xi)
        """
        mu, sigma = self.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        if acq_type.lower() == "ei":
            # Minimization EI
            improvement = (self.y_min - mu - xi)
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(0.0, ei)
        elif acq_type.lower() == "lcb":
            # Lower confidence bound for minimization (negated so higher is better)
            return -(mu - beta * sigma)
        elif acq_type.lower() == "pi":
            z = (self.y_min - mu - xi) / sigma
            return norm.cdf(z)
        else:
            raise ValueError(f"Unknown acquisition type: {acq_type}")


class CandidatePoolGenerator:
    """
    Generates a shared, broad candidate pool:
    C_t = C_acq U C_global U C_diverse
    """
    def __init__(self, bounds: np.ndarray, pool_size: int = 1000, 
                 ratio_acq: float = 0.4, ratio_global: float = 0.4, ratio_diverse: float = 0.2,
                 random_state: int = 42):
        self.bounds = np.array(bounds, dtype=float)
        self.dim = bounds.shape[0]
        self.pool_size = pool_size
        self.ratio_acq = ratio_acq
        self.ratio_global = ratio_global
        self.ratio_diverse = ratio_diverse
        self.rng = np.random.RandomState(random_state)

    def generate(self, surrogate: Optional[TargetGPSurrogate] = None, 
                 current_X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate candidate pool C_t of shape (pool_size, dim).
        """
        n_global = int(self.pool_size * self.ratio_global)
        n_diverse = int(self.pool_size * self.ratio_diverse)
        n_acq = self.pool_size - n_global - n_diverse
        
        candidates = []
        
        # 1. Global uniform exploration
        c_global = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_global, self.dim))
        candidates.append(c_global)
        
        # 2. Diverse perturbations around current evaluated points
        if current_X is not None and len(current_X) > 0:
            idx_sampled = self.rng.choice(len(current_X), size=n_diverse, replace=True)
            scales = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
            noise = self.rng.normal(0, scales, size=(n_diverse, self.dim))
            c_diverse = current_X[idx_sampled] + noise
            # Clip to bounds
            c_diverse = np.clip(c_diverse, self.bounds[:, 0], self.bounds[:, 1])
        else:
            c_diverse = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_diverse, self.dim))
        candidates.append(c_diverse)
        
        # 3. Acquisition-focused candidates (dense uniform screening or gradient local starts)
        # We sample a large batch and keep highest acquisition candidates if surrogate is available
        if surrogate is not None and surrogate.is_fitted:
            screening_batch = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_acq * 5, self.dim))
            acq_scores = surrogate.compute_acquisition(screening_batch, acq_type="ei")
            top_acq_idx = np.argsort(acq_scores)[-n_acq:]
            c_acq = screening_batch[top_acq_idx]
        else:
            c_acq = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_acq, self.dim))
        candidates.append(c_acq)
        
        pool = np.vstack(candidates)
        # Final safety clip
        pool = np.clip(pool, self.bounds[:, 0], self.bounds[:, 1])
        return pool
