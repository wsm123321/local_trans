"""
Sequential Bayesian Optimization Loop with Region Reranking.
Implements closed-loop iterative optimization:
Each comparator maintains its own target surrogate, generates candidates,
selects points, evaluates the target landscape, and updates its observations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .surrogate_and_candidates import TargetGPSurrogate, CandidatePoolGenerator
from .rerankers import (
    BaseReranker, TargetOnlyReranker, SoftRegionReranker, 
    HardFilterReranker, TrueOracleReranker
)
from .source_regions import SourceRegionLibrary


class SequentialBOEngine:
    """
    Closed-loop sequential Bayesian Optimization runner for a single strategy.
    """
    def __init__(self, target_func, bounds: np.ndarray, 
                 reranker: BaseReranker, pool_size: int = 500,
                 lambda_decay: float = 0.05,
                 rng: Optional[np.random.Generator] = None):
        self.target_func = target_func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = bounds.shape[0]
        self.reranker = reranker
        self.pool_size = pool_size
        self.lambda_decay = float(lambda_decay)
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def optimize(self, init_X: np.ndarray, init_y: np.ndarray, 
                 budget: int = 15) -> Dict[str, Any]:
        """
        Run sequential BO starting from (init_X, init_y) for `budget` iterations.
        """
        X_history = list(init_X.copy())
        y_history = list(init_y.copy())
        
        current_best_y = float(np.min(y_history))
        best_y_trace = [current_best_y]
        
        for t in range(1, budget + 1):
            curr_X_arr = np.array(X_history)
            curr_y_arr = np.array(y_history)
            
            # 1. Fit Target GP
            surrogate = TargetGPSurrogate(dim=self.dim, random_state=int(self.rng.integers(0, 100000)))
            surrogate.fit(curr_X_arr, curr_y_arr)
            
            # 2. Generate candidate pool
            pool_gen = CandidatePoolGenerator(bounds=self.bounds, pool_size=self.pool_size, rng=self.rng)
            candidates = pool_gen.generate(surrogate=surrogate, current_X=curr_X_arr)
            
            # 3. Compute target acquisition
            acq_scores = surrogate.compute_acquisition(candidates, acq_type="ei")
            
            # 4. Adaptive lambda annealing if reranker is SoftRegionReranker
            if isinstance(self.reranker, SoftRegionReranker):
                base_lambda = 1.0
                curr_lambda = base_lambda / (1.0 + self.lambda_decay * t)
                self.reranker.weight_lambda = curr_lambda
                
            # 5. Score and rank candidates
            ranked_idx, _ = self.reranker.score_and_rank(candidates, acq_scores)
            selected_x = candidates[ranked_idx[0]]
            
            # 6. Evaluate selected candidate on true target landscape
            selected_y = float(self.target_func(selected_x.reshape(1, -1))[0])
            
            # 7. Update dataset and best record
            X_history.append(selected_x)
            y_history.append(selected_y)
            
            current_best_y = min(current_best_y, selected_y)
            best_y_trace.append(current_best_y)
            
        return {
            "method": self.reranker.name,
            "final_best_y": current_best_y,
            "best_y_trace": best_y_trace,
            "all_evaluated_y": y_history,
        }
