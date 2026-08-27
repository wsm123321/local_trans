"""
Synthetic and Benchmark Optimization Landscapes for Region-Guided Transfer Verification.
Provides controllable multi-modal landscapes with known ground-truth basin structures,
controllable source-target shift, rotation, and deformation.
"""

from typing import Dict, List, Tuple, Optional, Union
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
                 rng: Optional[np.random.Generator] = None):
        super().__init__(dim=dim, bounds=bounds, name=f"GMM_{dim}D")
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.noise_std = float(noise_std)
        
        if centers is None:
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
            self.centers = [np.array(c, dtype=float).ravel() for c in centers]
            self.covs = [np.array(cov, dtype=float) for cov in covs] if covs is not None else [np.eye(dim) for _ in centers]
            self.weights = list(weights) if weights is not None else [1.0] * len(self.centers)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        vals = np.zeros(N)
        for k, (mu, sigma, w) in enumerate(zip(self.centers, self.covs, self.weights)):
            inv_sig = np.linalg.inv(sigma + 1e-5 * np.eye(d))
            diff = X - mu[np.newaxis, :]
            quad = np.sum((diff @ inv_sig) * diff, axis=1)
            vals += w * np.exp(-0.5 * np.maximum(0.0, quad))
        y = -vals
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        basins = []
        max_w = max(self.weights)
        for mu, cov, w in zip(self.centers, self.covs, self.weights):
            basins.append({
                "center": mu.copy(),
                "cov": cov.copy(),
                "weight": float(w),
                "is_global": (w == max_w)
            })
        return basins


class ShiftedRotatedRastrigin(BaseLandscape):
    """
    Rastrigin landscape with shift and orthogonal rotation.
    f(x) = 10*d + sum(z_i^2 - 10*cos(2*pi*z_i)), where z = R * (x - x_opt)
    """
    def __init__(self, dim: int = 2, shift: Optional[np.ndarray] = None, 
                 rotation_angle: float = 0.0, noise_std: float = 0.0, 
                 rng: Optional[np.random.Generator] = None):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.12
        bounds[:, 1] = 5.12
        super().__init__(dim=dim, bounds=bounds, name=f"Rastrigin_{dim}D")
        self.shift = np.zeros(dim) if shift is None else np.array(shift, dtype=float).ravel()
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.noise_std = float(noise_std)
        
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
    One basin (mu1) contains the global optimum, the second (mu2) contains a deceptive local optimum.
    """
    def __init__(self, dim: int = 2, mu1: Optional[np.ndarray] = None, 
                 mu2: Optional[np.ndarray] = None, d_scale: float = 1.0, 
                 noise_std: float = 0.0, rng: Optional[np.random.Generator] = None):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.0
        bounds[:, 1] = 5.0
        super().__init__(dim=dim, bounds=bounds, name=f"Lunacek_{dim}D")
        
        if mu1 is not None:
            self.mu1 = np.array(mu1, dtype=float).ravel()
        else:
            self.mu1 = np.ones(dim) * 2.5
            
        if mu2 is not None:
            self.mu2 = np.array(mu2, dtype=float).ravel()
        else:
            self.mu2 = -self.mu1.copy()
            
        self.d_scale = float(d_scale)
        self.noise_std = float(noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.s = 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N, d = X.shape
        d1 = np.sum((X - self.mu1[np.newaxis, :])**2, axis=1)
        d2 = self.d_scale * d + self.s * np.sum((X - self.mu2[np.newaxis, :])**2, axis=1)
        
        ras = 10.0 * (d - np.sum(np.cos(2.0 * np.pi * (X - self.mu1[np.newaxis, :])), axis=1))
        y = np.minimum(d1, d2) + ras
        if self.noise_std > 0:
            y += self.rng.normal(0, self.noise_std, size=N)
        return y

    def get_oracle_basins(self) -> List[Dict]:
        return [
            {"center": self.mu1.copy(), "cov": np.eye(self.dim) * 0.3, "weight": 1.0, "is_global": True},
            {"center": self.mu2.copy(), "cov": np.eye(self.dim) * 0.5, "weight": 0.7, "is_global": False}
        ]


class ShiftedAckley(BaseLandscape):
    """Ackley multimodal benchmark function with controllable shift."""
    def __init__(self, dim: int = 2, shift: Optional[np.ndarray] = None, 
                 noise_std: float = 0.0, rng: Optional[np.random.Generator] = None):
        bounds = np.zeros((dim, 2))
        bounds[:, 0] = -5.0
        bounds[:, 1] = 5.0
        super().__init__(dim=dim, bounds=bounds, name=f"Ackley_{dim}D")
        self.shift = np.zeros(dim) if shift is None else np.array(shift, dtype=float).ravel()
        self.noise_std = float(noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(42)

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


def get_task_suite(dim: int = 2, rng: Optional[np.random.Generator] = None) -> Dict[str, Dict]:
    """
    Returns a dictionary of problem suites for controlled transfer verification.
    Each problem suite contains:
      - target_func: The evaluation landscape
      - matching_sources: List of historical source tasks with shared/slightly shifted good regions
      - mismatched_sources: List of source tasks with distinct/conflicting optimal regions
      - bounds: Landscape bounds
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    suite = {}
    
    # 1. GMM Landscape
    target_gmm = GaussianMixtureLandscape(dim=dim, rng=rng)
    matching_gmm = []
    for i in range(3):
        perturbed_centers = [c + rng.normal(0, 0.15, size=dim) for c in target_gmm.centers]
        src = GaussianMixtureLandscape(dim=dim, centers=perturbed_centers, 
                                       covs=target_gmm.covs, weights=target_gmm.weights, rng=rng)
        matching_gmm.append(src)
        
    mismatched_gmm = []
    for i in range(3):
        # Displaced centers inside bounds and inverted weights (so global basin becomes local)
        displaced_centers = [-c for c in target_gmm.centers]
        src = GaussianMixtureLandscape(dim=dim, centers=displaced_centers, 
                                       covs=target_gmm.covs, weights=target_gmm.weights[::-1], rng=rng)
        mismatched_gmm.append(src)
        
    suite["GMM"] = {
        "target": target_gmm,
        "matching_sources": matching_gmm,
        "mismatched_sources": mismatched_gmm,
        "bounds": target_gmm.bounds
    }
    
    # 2. Shifted Rastrigin Landscape
    target_shift = rng.uniform(-1.5, 1.5, size=dim)
    target_rastrigin = ShiftedRotatedRastrigin(dim=dim, shift=target_shift, rotation_angle=0.1, rng=rng)
    matching_rastrigin = [
        ShiftedRotatedRastrigin(dim=dim, shift=target_shift + rng.normal(0, 0.15, size=dim), rotation_angle=0.12, rng=rng),
        ShiftedRotatedRastrigin(dim=dim, shift=target_shift + rng.normal(0, 0.25, size=dim), rotation_angle=0.08, rng=rng),
    ]
    mismatched_rastrigin = [
        ShiftedRotatedRastrigin(dim=dim, shift=-target_shift, rotation_angle=0.8, rng=rng),
        ShiftedRotatedRastrigin(dim=dim, shift=np.clip(target_shift + 3.0, -4.5, 4.5), rotation_angle=1.2, rng=rng),
    ]
    suite["Rastrigin"] = {
        "target": target_rastrigin,
        "matching_sources": matching_rastrigin,
        "mismatched_sources": mismatched_rastrigin,
        "bounds": target_rastrigin.bounds
    }
    
    # 3. Lunacek Bi-Rastrigin Landscape
    target_mu1 = rng.uniform(1.8, 2.5, size=dim)
    target_luna = LunacekBiRastrigin(dim=dim, mu1=target_mu1, rng=rng)
    matching_luna = [
        LunacekBiRastrigin(dim=dim, mu1=target_mu1 + rng.normal(0, 0.15, size=dim), rng=rng),
        LunacekBiRastrigin(dim=dim, mu1=target_mu1 + rng.normal(0, 0.25, size=dim), rng=rng),
    ]
    mismatched_luna = [
        LunacekBiRastrigin(dim=dim, mu1=-target_mu1, mu2=target_mu1, rng=rng),
        LunacekBiRastrigin(dim=dim, mu1=-target_mu1 + rng.normal(0, 0.2, size=dim), rng=rng)
    ]
    suite["Lunacek"] = {
        "target": target_luna,
        "matching_sources": matching_luna,
        "mismatched_sources": mismatched_luna,
        "bounds": target_luna.bounds
    }
    
    # 4. Shifted Ackley Landscape
    target_ack_shift = rng.uniform(-1.2, 1.2, size=dim)
    target_ackley = ShiftedAckley(dim=dim, shift=target_ack_shift, rng=rng)
    matching_ackley = [
        ShiftedAckley(dim=dim, shift=target_ack_shift + rng.normal(0, 0.15, size=dim), rng=rng),
        ShiftedAckley(dim=dim, shift=target_ack_shift + rng.normal(0, 0.25, size=dim), rng=rng),
    ]
    mismatched_ackley = [
        ShiftedAckley(dim=dim, shift=-target_ack_shift, rng=rng),
        ShiftedAckley(dim=dim, shift=np.clip(target_ack_shift + 3.0, -4.5, 4.5), rng=rng),
    ]
    suite["Ackley"] = {
        "target": target_ackley,
        "matching_sources": matching_ackley,
        "mismatched_sources": mismatched_ackley,
        "bounds": target_ackley.bounds
    }
    
    return suite
