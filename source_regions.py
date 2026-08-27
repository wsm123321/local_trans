"""
Source Region Extraction, Modeling, and Support Scoring Module.
Implements:
1. Extraction of high-quality regions from historical source observations.
2. Parameter estimation: (mu_k, Sigma_k, q_k, n_k).
3. Cross-task region library aggregation and deduplication.
4. Support score evaluation r_s(x) for target candidate points.
5. Synthetic region generators: Random-Region, Wrong-Source, and Oracle-Target-Region.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class SourceRegion:
    """
    Representation of an individual high-quality source region.
    R_k = (mu_k, Sigma_k, q_k, n_k)
    """
    def __init__(self, center: np.ndarray, cov: np.ndarray, quality: float, 
                 count: int = 1, source_task_id: Optional[str] = None):
        self.center = np.array(center, dtype=float)
        self.cov = np.array(cov, dtype=float)
        self.quality = float(quality)  # q_k in [0, 1]
        self.count = int(count)        # n_k (evidence weight / sample size)
        self.source_task_id = source_task_id
        
        # Precompute inverse and determinant for fast Mahalanobis scoring
        dim = len(self.center)
        # Adaptive regularization to prevent ill-conditioned or vanishing bandwidth in higher dimensions
        min_var = 1e-2
        diag_floor = np.maximum(np.diag(self.cov), min_var)
        reg_cov = self.cov.copy()
        np.fill_diagonal(reg_cov, np.maximum(np.diag(reg_cov), diag_floor))
        reg_cov += 1e-3 * np.eye(dim)
        
        try:
            self.inv_cov = np.linalg.inv(reg_cov)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(reg_cov)
            
    def compute_support(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian support score:
        s(x) = q_k * exp(-0.5 * (x - mu_k)^T (Sigma_k + eps I)^-1 (x - mu_k) / sqrt(d))
        X: shape (N, d)
        Returns: 1D array of length N
        """
        X = np.atleast_2d(X)
        diff = X - self.center[np.newaxis, :]
        dim = X.shape[1]
        quad = np.sum((diff @ self.inv_cov) * diff, axis=1)
        # Dimension scaling to maintain gradient in higher dimensions
        scaled_quad = np.maximum(0.0, quad) / np.sqrt(max(1.0, float(dim)))
        return self.quality * np.exp(-0.5 * scaled_quad)


class SourceRegionLibrary:
    """
    Collection of Source Regions R_s = {R_1, ..., R_K}.
    Computes aggregated support score r_s(x) = max_k s_k(x) (or soft max).
    """
    def __init__(self, regions: Optional[List[SourceRegion]] = None):
        self.regions = list(regions) if regions is not None else []
        
    def add_region(self, region: SourceRegion):
        self.regions.append(region)
        
    def score(self, X: np.ndarray, aggregation: str = "max") -> np.ndarray:
        """
        Compute r_s(x) across all regions in library.
        If library is empty, returns zeros.
        """
        X = np.atleast_2d(X)
        N = X.shape[0]
        if not self.regions:
            return np.zeros(N)
            
        region_scores = np.zeros((len(self.regions), N))
        for i, reg in enumerate(self.regions):
            region_scores[i, :] = reg.compute_support(X)
            
        if aggregation == "max":
            return np.max(region_scores, axis=0)
        elif aggregation == "weighted_sum":
            # Weight by region count n_k
            weights = np.array([r.count for r in self.regions], dtype=float)
            weights /= np.sum(weights)
            return np.sum(region_scores * weights[:, np.newaxis], axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


class SourceRegionExtractor:
    """
    Extracts high-quality regions from source task datasets.
    """
    def __init__(self, top_ratio: float = 0.20, max_clusters: int = 3, 
                 min_samples_per_cluster: int = 3, random_state: int = 42):
        self.top_ratio = top_ratio
        self.max_clusters = max_clusters
        self.min_samples_per_cluster = min_samples_per_cluster
        self.random_state = random_state

    def extract_from_dataset(self, X: np.ndarray, y: np.ndarray, 
                              task_id: str = "source_0") -> List[SourceRegion]:
        """
        Extract regions from a single source dataset (X: N x d, y: N).
        Assumes minimization problem (lower y is better).
        """
        N, d = X.shape
        if N < 5:
            return []
            
        # 1. Rank-normalization of target values y within task
        ranks = np.argsort(np.argsort(y))  # 0 is best
        norm_quality = 1.0 - (ranks / (N - 1.0 + 1e-8))  # in [0, 1], 1 is best
        
        # 2. Select top_ratio high quality samples
        k_top = max(self.min_samples_per_cluster, int(np.ceil(N * self.top_ratio)))
        top_idx = np.argsort(y)[:k_top]
        
        X_top = X[top_idx]
        q_top = norm_quality[top_idx]
        
        if len(X_top) < self.min_samples_per_cluster:
            return []
            
        # 3. Cluster top samples in decision space
        n_clusters = min(self.max_clusters, max(1, len(X_top) // self.min_samples_per_cluster))
        
        regions = []
        if n_clusters == 1:
            mu = np.mean(X_top, axis=0)
            cov = np.cov(X_top.T) if d > 1 else np.var(X_top) * np.eye(1)
            if d > 1 and cov.ndim < 2:
                cov = np.eye(d) * 0.5
            elif d == 1:
                cov = np.atleast_2d(cov)
            q_mean = float(np.mean(q_top))
            regions.append(SourceRegion(mu, cov, quality=q_mean, count=len(X_top), source_task_id=task_id))
        else:
            try:
                gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', 
                                      random_state=self.random_state, reg_covar=1e-3)
                labels = gmm.fit_predict(X_top)
                
                for c in range(n_clusters):
                    c_mask = (labels == c)
                    if np.sum(c_mask) < 2:
                        continue
                    mu = gmm.means_[c]
                    cov = gmm.covariances_[c]
                    q_c = float(np.mean(q_top[c_mask]))
                    count_c = int(np.sum(c_mask))
                    regions.append(SourceRegion(mu, cov, quality=q_c, count=count_c, source_task_id=task_id))
            except Exception:
                # Fallback to single cluster
                mu = np.mean(X_top, axis=0)
                cov = np.cov(X_top.T) if d > 1 else np.var(X_top) * np.eye(1)
                cov = np.atleast_2d(cov) if d == 1 else cov
                q_mean = float(np.mean(q_top))
                regions.append(SourceRegion(mu, cov, quality=q_mean, count=len(X_top), source_task_id=task_id))
                
        return regions

    def extract_from_multi_sources(self, source_datasets: List[Tuple[np.ndarray, np.ndarray]], 
                                   task_ids: Optional[List[str]] = None) -> SourceRegionLibrary:
        """
        Extract regions across multiple source tasks and assemble the library.
        """
        library = SourceRegionLibrary()
        for idx, (X, y) in enumerate(source_datasets):
            tid = task_ids[idx] if task_ids and idx < len(task_ids) else f"src_{idx}"
            task_regs = self.extract_from_dataset(X, y, task_id=tid)
            for r in task_regs:
                library.add_region(r)
        return library


def create_random_region_library(dim: int, bounds: np.ndarray, num_regions: int = 3, 
                                 seed: int = 123) -> SourceRegionLibrary:
    """Generate fake random regions uniformly in the search space."""
    rng = np.random.RandomState(seed)
    lib = SourceRegionLibrary()
    for i in range(num_regions):
        mu = rng.uniform(bounds[:, 0], bounds[:, 1])
        # Random spherical/elliptical covariance
        scales = rng.uniform(0.2, 0.8, size=dim)
        cov = np.diag(scales**2)
        q = rng.uniform(0.7, 1.0)
        lib.add_region(SourceRegion(mu, cov, quality=q, count=5, source_task_id="random_fake"))
    return lib


def create_oracle_region_library(oracle_basins: List[Dict], dim: int) -> SourceRegionLibrary:
    """Create Oracle region library directly from ground truth target landscape optima."""
    lib = SourceRegionLibrary()
    for i, b in enumerate(oracle_basins):
        mu = b["center"]
        cov = b.get("cov", np.eye(dim) * 0.3)
        w = b.get("weight", 1.0)
        lib.add_region(SourceRegion(mu, cov, quality=w, count=20, source_task_id="oracle_target"))
    return lib
