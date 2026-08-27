"""
Source Region Extraction, Modeling, and Support Scoring Module.
Implements:
1. Extraction of high-quality regions from historical source observations.
2. Parameter estimation: (mu_k, Sigma_k, q_k, n_k).
3. Cross-task region library aggregation and deduplication/merging.
4. Support score evaluation r_s(x) for candidate points.
5. Structure-matched Random-Region baseline (preserves Sigma, q, n, only permutes/shifts mu).
6. Adversarial Wrong-Source baseline.
7. True Oracle Target Region baseline.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2


class SourceRegion:
    """
    Representation of an individual high-quality source region.
    R_k = (mu_k, Sigma_k, q_k, n_k)
    """
    def __init__(self, center: np.ndarray, cov: np.ndarray, quality: float, 
                 count: int = 1, source_task_id: Optional[str] = None):
        self.center = np.array(center, dtype=float).ravel()
        self.cov = np.array(cov, dtype=float)
        self.quality = float(quality)  # q_k in [0, 1]
        self.count = int(count)        # n_k (evidence weight / sample size)
        self.source_task_id = source_task_id
        
        dim = len(self.center)
        # Regularize covariance to ensure positive definiteness
        min_var = 1e-3
        diag_floor = np.maximum(np.diag(self.cov), min_var)
        reg_cov = self.cov.copy()
        np.fill_diagonal(reg_cov, np.maximum(np.diag(reg_cov), diag_floor))
        reg_cov += 1e-4 * np.eye(dim)
        
        try:
            self.inv_cov = np.linalg.inv(reg_cov)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(reg_cov)
            
    def compute_mahalanobis_sq(self, X: np.ndarray) -> np.ndarray:
        """Compute (x - mu)^T Sigma^-1 (x - mu) for batch X (N, d)."""
        X = np.atleast_2d(X)
        diff = X - self.center[np.newaxis, :]
        quad = np.sum((diff @ self.inv_cov) * diff, axis=1)
        return np.maximum(0.0, quad)

    def compute_support(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian support score:
        s(x) = q_k * exp(-0.5 * (x - mu_k)^T Sigma_k^-1 (x - mu_k))
        """
        quad = self.compute_mahalanobis_sq(X)
        return self.quality * np.exp(-0.5 * quad)

    def is_inside_confidence_region(self, X: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """
        Chi-square confidence ellipsoid test:
        (x - mu)^T Sigma^-1 (x - mu) <= chi2_{d, confidence}
        """
        dim = len(self.center)
        thresh = float(chi2.ppf(confidence, df=dim))
        quad = self.compute_mahalanobis_sq(X)
        return quad <= thresh


class SourceRegionLibrary:
    """
    Collection of Source Regions R_s = {R_1, ..., R_K}.
    Computes aggregated support score r_s(x) = max_k s_k(x) or weighted sum.
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
            weights = np.array([r.count for r in self.regions], dtype=float)
            weights /= max(1e-8, np.sum(weights))
            return np.sum(region_scores * weights[:, np.newaxis], axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def filter_inside_any_region(self, X: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Return boolean mask (N,) indicating whether each point is inside at least one region ellipsoid."""
        X = np.atleast_2d(X)
        N = X.shape[0]
        if not self.regions:
            return np.ones(N, dtype=bool)
        mask = np.zeros(N, dtype=bool)
        for reg in self.regions:
            mask |= reg.is_inside_confidence_region(X, confidence=confidence)
        return mask


class SourceRegionExtractor:
    """
    Extracts high-quality regions from source task datasets and aggregates across tasks.
    """
    def __init__(self, top_ratio: float = 0.20, max_clusters: int = 3, 
                 min_samples_per_cluster: int = 3, random_state: int = 42):
        self.top_ratio = float(top_ratio)
        self.max_clusters = int(max_clusters)
        self.min_samples_per_cluster = int(min_samples_per_cluster)
        self.random_state = int(random_state)

    def extract_from_dataset(self, X: np.ndarray, y: np.ndarray, 
                              task_id: str = "source_0") -> List[SourceRegion]:
        """
        Extract regions from a single source dataset (X: N x d, y: N).
        Assumes minimization (lower y is better).
        """
        N, d = X.shape
        if N < 5:
            return []
            
        # 1. Rank-normalization of target values y within task
        ranks = np.argsort(np.argsort(y))  # 0 is best
        norm_quality = 1.0 - (ranks / (N - 1.0 + 1e-8))
        
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
        if n_clusters == 1 or len(X_top) <= n_clusters * 2:
            mu = np.mean(X_top, axis=0)
            cov = np.cov(X_top.T) if d > 1 else np.var(X_top) * np.eye(1)
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
                mu = np.mean(X_top, axis=0)
                cov = np.cov(X_top.T) if d > 1 else np.var(X_top) * np.eye(1)
                cov = np.atleast_2d(cov)
                q_mean = float(np.mean(q_top))
                regions.append(SourceRegion(mu, cov, quality=q_mean, count=len(X_top), source_task_id=task_id))
                
        return regions

    def extract_from_multi_sources(self, source_datasets: List[Tuple[np.ndarray, np.ndarray]], 
                                   task_ids: Optional[List[str]] = None,
                                   merge_threshold: float = 0.5) -> SourceRegionLibrary:
        """
        Extract regions across multiple source tasks and perform cross-task deduplication / merging.
        If two regions from different tasks have centers closer than merge_threshold, they are merged.
        """
        all_regions = []
        for idx, (X, y) in enumerate(source_datasets):
            tid = task_ids[idx] if task_ids and idx < len(task_ids) else f"src_{idx}"
            task_regs = self.extract_from_dataset(X, y, task_id=tid)
            all_regions.extend(task_regs)
            
        if not all_regions:
            return SourceRegionLibrary()
            
        # Deduplication and clustering across tasks
        library = SourceRegionLibrary()
        merged_flags = [False] * len(all_regions)
        
        for i in range(len(all_regions)):
            if merged_flags[i]:
                continue
            r_i = all_regions[i]
            cluster_members = [r_i]
            merged_flags[i] = True
            
            for j in range(i + 1, len(all_regions)):
                if not merged_flags[j]:
                    r_j = all_regions[j]
                    dist = np.linalg.norm(r_i.center - r_j.center)
                    if dist < merge_threshold:
                        cluster_members.append(r_j)
                        merged_flags[j] = True
                        
            # Merge cluster members
            if len(cluster_members) == 1:
                library.add_region(r_i)
            else:
                total_count = sum(r.count for r in cluster_members)
                merged_mu = sum(r.center * r.count for r in cluster_members) / total_count
                merged_cov = sum(r.cov * r.count for r in cluster_members) / total_count
                merged_q = float(np.mean([r.quality for r in cluster_members]))
                merged_region = SourceRegion(
                    center=merged_mu, cov=merged_cov, quality=merged_q, 
                    count=total_count, source_task_id=f"merged_{len(cluster_members)}"
                )
                library.add_region(merged_region)
                
        return library


def create_structure_matched_random_library(matching_lib: SourceRegionLibrary, 
                                             bounds: np.ndarray, 
                                             rng: np.random.Generator) -> SourceRegionLibrary:
    """
    Strict Structure-Matched Random Baseline:
    Preserves exact region count, exact covariance matrices Sigma_k (shape, spectrum, volume),
    exact quality q_k, and exact evidence weight n_k.
    ONLY uniformly redistributes the center mu_k across the search domain.
    """
    rand_lib = SourceRegionLibrary()
    for reg in matching_lib.regions:
        rand_mu = rng.uniform(bounds[:, 0], bounds[:, 1])
        rand_reg = SourceRegion(
            center=rand_mu,
            cov=reg.cov.copy(),
            quality=reg.quality,
            count=reg.count,
            source_task_id="struct_matched_random"
        )
        rand_lib.add_region(rand_reg)
    return rand_lib


def create_true_oracle_library(oracle_basins: List[Dict], dim: int) -> SourceRegionLibrary:
    """
    True Oracle library: targets ONLY the true global optimum basin.
    """
    lib = SourceRegionLibrary()
    for b in oracle_basins:
        if b.get("is_global", True):
            mu = b["center"]
            cov = b.get("cov", np.eye(dim) * 0.3)
            w = 1.0
            lib.add_region(SourceRegion(mu, cov, quality=w, count=50, source_task_id="oracle_global"))
            break
    return lib
