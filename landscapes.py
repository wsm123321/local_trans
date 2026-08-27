"""
Synthetic & Benchmark Problem Landscapes for Region-Guided Transfer Verification.
Provides controllable multi-modal landscapes with known ground truth basins,
controllable source-target shift, rotation, and deformation.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np


class BaseLandscape:
    """Base class for optimization landscapes."""
    def __init__(self, dim: int = 2, bounds: Optional[np.ndarray] = None, name: str = "BaseLandscape"):
        self.dim = dim
        self.name = name
        if bounds is None:
            self.bounds = np.zeros((dim, 2))
            self.bounds[:, 0] = -5.0
            self.bounds[:, 1] = 5.0
        else:
            self.bounds = np.array(bounds, dtype=float)
            
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate landscape at X (N, d). Returns 1D array of length N."""
        raise NotImplementedError
        
    def get_oracle_basins(self) -> List[Dict]:
        """Return list of dicts with true optimal / sub-optimal basin centers and properties."""
        return []


class GaussianMixtureLandscape(BaseLandscape):
    """
    Controllable Gaussian mixture potential landscape.
    f(x) = - sum_{k} w_k * exp(- 0.5 * (x - mu_k)^T Sigma_k^-1 (x - mu_k))
    Minimization problem: lowest f(x) corresponds to the highest peak.
    """
    def __init__(self, dim: int = 2, bounds: Optional[np.ndarray] = None, 
                 centers: Optional[List[np.ndarray]] = None,
                 covs: Optional[List[np.ndarray]] = None,
                 weights: Optional[List[float]] = None,
                 noise_std: float = 0.0,
                 seed: int = 42):
        super().__init__(dim=dim, bounds=bounds, name=f"GMM_{dim}D")
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        
        if centers is None:
            # Generate 4 distinct basins
            num_basins = 4
            self.centers = []
            self.covs = []
            self.weights = [1.0, 0.8, 0.6, 0.5]
            
            # Global minimum near (1.5, 1.5, ...)
            c1 = np.ones(dim) * 1.5
            self.centers.append(c1)
            self.covs.append(np.eye(dim) * 0.8)
            
            # Sub-optima
            c2 = -np.ones(dim) * 2.0
            self.centers.append(c2)
            self.covs.append(np.eye(dim) * 1.2)
            
            c3 = np.zeros(dim)
            c3[0] = 2.5
            if dim > 1:
                c3[1] = -2.5
            self.centers.append(c3)
            self.covs.append(np.eye(dim) * 1.0)
            
            c4 = np.zeros(dim)
            c4[0] = -2.5
            if dim > 1:
                c4[1] = 2.5
            self.centers.append(c4)
            self.covs.append(np.eye(dim) * 0.9)
        else:
            self.centers = [np.array(c, dtype=float) for c in centers]
            self.covs = [np.array(cov, dtype=float) for cov in covs] if covs is not None else [np.eye(dim) for _ in centers]
            self.weights = list(weights) if weights is not None else [1.0] * len(self.centers)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        vals = np.zeros(N)
        for k, (mu, sigma, w) in enumerate(zip(self.centers, self.covs, self.weights)):
            inv_sig = np.linalg.inv(sigma + 1e-6 * np.eye(d))
            diff = X - mu[np.newaxis, :]
            # Mahalanobis dist sq
            quad = np.sum((diff @ inv_sig) * diff, axis=1)
            vals += w * np.exp(-0.5 * quad)
        # Minimize: negative peak value
        y = -vals
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        basins = []
        for mu, cov, w in zip(self.centers, self.covs, self.weights):
            basins.append({
                "center": mu.copy(),
                "cov": cov.copy(),
                "weight": w,
                "is_global": (w == max(self.weights))
            })
        return basins


class ShiftedRotatedRastrigin(BaseLandscape):
    """
    Rastrigin landscape with optional shift and orthogonal rotation.
    f(x) = 10*d + sum(z_i^2 - 10*cos(2*pi*z_i)), where z = R * (x - x_opt)
    """
    def __init__(self, dim: int = 2, shift: Optional[np.ndarray] = None, 
                 rotation_angle: float = 0.0, noise_std: float = 0.0, seed: int = 42):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.12
        bounds[:, 1] = 5.12
        super().__init__(dim=dim, bounds=bounds, name=f"Rastrigin_{dim}D")
        self.shift = np.zeros(dim) if shift is None else np.array(shift, dtype=float)
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        
        # Orthogonal rotation matrix
        self.R = np.eye(dim)
        if dim >= 2 and abs(rotation_angle) > 1e-5:
            theta = rotation_angle
            c, s = np.cos(theta), np.sin(theta)
            self.R[0, 0] = c
            self.R[0, 1] = -s
            self.R[1, 0] = s
            self.R[1, 1] = c

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        Z = (X - self.shift[np.newaxis, :]) @ self.R.T
        y = 10.0 * d + np.sum(Z**2 - 10.0 * np.cos(2.0 * np.pi * Z), axis=1)
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        return [{
            "center": self.shift.copy(),
            "cov": np.eye(self.dim) * 0.2,
            "weight": 1.0,
            "is_global": True
        }]


class LunacekBiRastrigin(BaseLandscape):
    """
    Lunacek Bi-Rastrigin landscape: contains two major basins at mu1 and mu2.
    One basin contains the global optimum, the second contains a deceptive local optimum.
    Tests whether source region prior correctly differentiates or guides search.
    """
    def __init__(self, dim: int = 2, mu1_val: float = 2.5, d_scale: float = 1.0, 
                 noise_std: float = 0.0, seed: int = 42):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.0
        bounds[:, 1] = 5.0
        super().__init__(dim=dim, bounds=bounds, name=f"Lunacek_{dim}D")
        self.mu1 = np.ones(dim) * mu1_val
        self.mu2 = -np.ones(dim) * np.sqrt((mu1_val**2 - d_scale) / d_scale) if d_scale > 0 else -np.ones(dim) * mu1_val
        self.d_scale = d_scale
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.s = 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
        self.mu0 = 2.5

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        # Distance to mu1 and mu2
        d1 = np.sum((X - self.mu0)**2, axis=1)
        d2 = self.d_scale * d + self.s * np.sum((X + self.mu0)**2, axis=1)
        
        ras = 10.0 * (d - np.sum(np.cos(2.0 * np.pi * (X - self.mu0)), axis=1))
        y = np.minimum(d1, d2) + ras
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        return [
            {"center": np.ones(self.dim) * self.mu0, "cov": np.eye(self.dim) * 0.3, "weight": 1.0, "is_global": True},
            {"center": -np.ones(self.dim) * self.mu0, "cov": np.eye(self.dim) * 0.5, "weight": 0.7, "is_global": False}
        ]


class ShiftedAckley(BaseLandscape):
    """Ackley multimodal benchmark function with controllable shift."""
    def __init__(self, dim: int = 2, shift: Optional[np.ndarray] = None, noise_std: float = 0.0, seed: int = 42):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.0
        bounds[:, 1] = 5.0
        super().__init__(dim=dim, bounds=bounds, name=f"Ackley_{dim}D")
        self.shift = np.zeros(dim) if shift is None else np.array(shift, dtype=float)
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        Z = X - self.shift[np.newaxis, :]
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(Z**2, axis=1)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * Z), axis=1))
        y = term1 + term2 + 20.0 + np.e
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        return [{
            "center": self.shift.copy(),
            "cov": np.eye(self.dim) * 0.4,
            "weight": 1.0,
            "is_global": True
        }]


def get_task_suite(dim: int = 2, seed: int = 42) -> Dict[str, Dict]:
    """
    Returns a dictionary of problem suites for controlled transfer verification.
    Each problem suite contains:
      - target_func: The evaluation landscape
      - matching_sources: List of historical source tasks with shared or slightly shifted good regions
      - mismatched_sources: List of source tasks with distinct/conflicting optimal regions
      - oracle_basins: Exact known optimal basins for benchmark evaluation
    """
    rng = np.random.RandomState(seed)
    suite = {}
    
    # 1. GMM Landscape
    target_gmm = GaussianMixtureLandscape(dim=dim, seed=seed)
    # Matching source: same landscape + small center perturbations (std=0.15)
    matching_gmm = []
    for i in range(3):
        perturbed_centers = [c + rng.normal(0, 0.15, size=dim) for c in target_gmm.centers]
        src = GaussianMixtureLandscape(dim=dim, centers=perturbed_centers, 
                                       covs=target_gmm.covs, weights=target_gmm.weights, seed=seed+10+i)
        matching_gmm.append(src)
    # Mismatched source: inverted weights or displaced centers (far away)
    mismatched_gmm = []
    for i in range(3):
        displaced_centers = [c + rng.uniform(2.5, 4.0, size=dim) * (1 if j%2==0 else -1) 
                             for j, c in enumerate(target_gmm.centers)]
        src = GaussianMixtureLandscape(dim=dim, centers=displaced_centers, 
                                       covs=target_gmm.covs, weights=target_gmm.weights[::-1], seed=seed+20+i)
        mismatched_gmm.append(src)
        
    suite["GMM"] = {
        "target": target_gmm,
        "matching_sources": matching_gmm,
        "mismatched_sources": mismatched_gmm,
        "bounds": target_gmm.bounds
    }
    
    # 2. Shifted Rastrigin Landscape
    target_shift = rng.uniform(-1.5, 1.5, size=dim)
    target_rastrigin = ShiftedRotatedRastrigin(dim=dim, shift=target_shift, rotation_angle=0.1, seed=seed)
    matching_rastrigin = [
        ShiftedRotatedRastrigin(dim=dim, shift=target_shift + rng.normal(0, 0.2, size=dim), rotation_angle=0.15, seed=seed+1),
        ShiftedRotatedRastrigin(dim=dim, shift=target_shift + rng.normal(0, 0.3, size=dim), rotation_angle=0.05, seed=seed+2),
    ]
    mismatched_rastrigin = [
        ShiftedRotatedRastrigin(dim=dim, shift=target_shift + 3.0, rotation_angle=0.8, seed=seed+3),
        ShiftedRotatedRastrigin(dim=dim, shift=-target_shift - 2.5, rotation_angle=1.2, seed=seed+4),
    ]
    suite["Rastrigin"] = {
        "target": target_rastrigin,
        "matching_sources": matching_rastrigin,
        "mismatched_sources": mismatched_rastrigin,
        "bounds": target_rastrigin.bounds
    }
    
    # 3. Lunacek Bi-Rastrigin Landscape
    target_luna = LunacekBiRastrigin(dim=dim, seed=seed)
    matching_luna = [
        LunacekBiRastrigin(dim=dim, mu1_val=2.3, seed=seed+1),
        LunacekBiRastrigin(dim=dim, mu1_val=2.7, seed=seed+2),
    ]
    # Mismatched puts high attraction on the deceptive basin
    mismatched_luna = [
        LunacekBiRastrigin(dim=dim, mu1_val=-2.5, seed=seed+3),
        ShiftedAckley(dim=dim, shift=np.ones(dim)*(-3.0), seed=seed+4)
    ]
    suite["Lunacek"] = {
        "target": target_luna,
        "matching_sources": matching_luna,
        "mismatched_sources": mismatched_luna,
        "bounds": target_luna.bounds
    }
    
    # 4. Shifted Ackley Landscape
    target_ack_shift = rng.uniform(-1.0, 1.0, size=dim)
    target_ackley = ShiftedAckley(dim=dim, shift=target_ack_shift, seed=seed)
    matching_ackley = [
        ShiftedAckley(dim=dim, shift=target_ack_shift + rng.normal(0, 0.25, size=dim), seed=seed+1),
        ShiftedAckley(dim=dim, shift=target_ack_shift + rng.normal(0, 0.25, size=dim), seed=seed+2),
    ]
    mismatched_ackley = [
        ShiftedAckley(dim=dim, shift=target_ack_shift + 3.5, seed=seed+3),
        ShiftedAckley(dim=dim, shift=-target_ack_shift - 3.0, seed=seed+4),
    ]
    suite["Ackley"] = {
        "target": target_ackley,
        "matching_sources": matching_ackley,
        "mismatched_sources": mismatched_ackley,
        "bounds": target_ackley.bounds
    }
    
    return suite
