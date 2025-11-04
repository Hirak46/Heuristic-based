"""
CERBERUS-AGG: Cluster-filter Ensemble-mix Reputation-weight Byzantine-Resilient Aggregation

A novel hybrid aggregation algorithm for federated learning that combines:
1. Multi-view anomaly detection (norm, cosine, Mahalanobis, clustering)
2. Ensemble of robust aggregators (Trimmed-Mean, Median, Geometric Median, Multi-Krum, AGR)
3. Adaptive reputation-weighted mixing with online learning
4. Trust region projection for stability

Copyright 2025. Research implementation for publication.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import trim_mean
from sklearn.covariance import MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CerberusAggregator:
    """
    CERBERUS-AGG: A multi-stage Byzantine-resilient aggregation algorithm.
    
    The algorithm operates in 8 stages per round:
    1. State initialization (reputation, covariance, ensemble weights)
    2. Multi-view pre-screening (norm, cosine, Mahalanobis, clustering)
    3. Coordinate sanitization (trimming, winsorization)
    4. Core robust estimators computation
    5. Ensemble mixture via online Hedge algorithm
    6. Reputation update based on conformity
    7. Final weighted aggregation with reputation
    8. State refresh for next round
    
    Parameters
    ----------
    num_clients : int
        Total number of clients in the federation
    malicious_fraction : float
        Expected fraction of malicious clients (for tuning thresholds)
    learning_rate : float, default=0.1
        Learning rate for Hedge algorithm ensemble weight updates
    reputation_decay : float, default=0.95
        Exponential decay factor for reputation scores
    norm_clip_scale : float, default=3.0
        Number of standard deviations for norm clipping threshold
    cosine_threshold : float, default=0.5
        Minimum cosine similarity for acceptance (adaptive if None)
    mahalanobis_threshold : float, default=3.0
        Maximum Mahalanobis distance for acceptance (chi-squared based)
    trim_ratio : float, default=0.1
        Fraction to trim from each tail for Trimmed-Mean
    krum_candidates : int, default=None
        Number of candidates for Multi-Krum (if None, uses n-f-2 rule)
    enable_clustering : bool, default=True
        Whether to use clustering-based anomaly detection
    enable_mahalanobis : bool, default=True
        Whether to use Mahalanobis distance (expensive for high-dim)
    validation_loss_fn : callable, default=None
        Function to compute validation loss for Hedge algorithm
        Signature: f(aggregated_params) -> float
    verbose : bool, default=False
        Whether to print detailed logs
    
    Attributes
    ----------
    reputation_scores_ : NDArray
        Current reputation score for each client [0, 1]
    ensemble_weights_ : NDArray
        Current ensemble weights for 5 core estimators (sum to 1)
    robust_covariance_ : Optional[MinCovDet]
        Robust covariance estimator for Mahalanobis distance
    historical_updates_ : List[NDArray]
        Buffer of recent updates for covariance estimation
    statistics_ : Dict
        Statistics collected during aggregation for analysis
    """
    
    def __init__(
        self,
        num_clients: int,
        malicious_fraction: float = 0.3,
        learning_rate: float = 0.1,
        reputation_decay: float = 0.95,
        norm_clip_scale: float = 3.0,
        cosine_threshold: Optional[float] = None,
        mahalanobobis_threshold: float = 3.0,
        trim_ratio: float = 0.1,
        krum_candidates: Optional[int] = None,
        enable_clustering: bool = True,
        enable_mahalanobis: bool = True,
        validation_loss_fn: Optional[callable] = None,
        verbose: bool = False,
    ):
        self.num_clients = num_clients
        self.malicious_fraction = malicious_fraction
        self.learning_rate = learning_rate
        self.reputation_decay = reputation_decay
        self.norm_clip_scale = norm_clip_scale
        self.cosine_threshold = cosine_threshold
        self.mahalanobis_threshold = mahalanobobis_threshold
        self.trim_ratio = trim_ratio
        self.krum_candidates = krum_candidates or max(1, num_clients - int(num_clients * malicious_fraction) - 2)
        self.enable_clustering = enable_clustering
        self.enable_mahalanobis = enable_mahalanobis
        self.validation_loss_fn = validation_loss_fn
        self.verbose = verbose
        
        # Initialize state
        self.reputation_scores_ = np.ones(num_clients)  # All clients start with full reputation
        self.ensemble_weights_ = np.ones(5) / 5.0  # Equal initial weights for 5 estimators
        self.robust_covariance_ = None
        self.historical_updates_ = []
        self.round_counter_ = 0
        
        # Statistics for publication metrics (store only scalar summaries, not full arrays)
        self.statistics_ = {
            'filtered_by_norm': [],
            'filtered_by_cosine': [],
            'filtered_by_mahalanobis': [],
            'filtered_by_clustering': [],
            'ensemble_losses': [],  # List of losses per round (small)
            'reputation_history': [],  # Will store only mean/min/max, not full array
            'selected_estimator': [],
        }
    
    def aggregate(
        self,
        client_updates: List[NDArray],
        client_ids: List[int],
        num_samples: Optional[List[int]] = None,
    ) -> Tuple[NDArray, Dict]:
        """
        Perform CERBERUS-AGG aggregation.
        
        Parameters
        ----------
        client_updates : List[NDArray]
            List of client parameter updates (each is a 1D numpy array)
        client_ids : List[int]
            Corresponding client IDs for reputation tracking
        num_samples : List[int], optional
            Number of samples each client trained on (for weighted aggregation)
        
        Returns
        -------
        aggregated_update : NDArray
            The final aggregated parameter update
        metadata : Dict
            Metadata about the aggregation process (for logging/analysis)
        """
        self.round_counter_ += 1
        n_participants = len(client_updates)
        
        if num_samples is None:
            num_samples = [1] * n_participants
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"CERBERUS-AGG Round {self.round_counter_}: {n_participants} participants")
            print(f"{'='*80}")
        
        # Convert list of arrays to matrix (each row is a client update)
        updates_matrix = np.array(client_updates)
        
        # ========================================================================
        # STAGE 1: State Initialization (already done in __init__ and maintained)
        # ========================================================================
        
        # ========================================================================
        # STAGE 2: Multi-view Pre-screening
        # ========================================================================
        survivors, filtered_metadata = self._multi_view_prescreening(
            updates_matrix, client_ids, num_samples
        )
        
        if len(survivors) == 0:
            warnings.warn("All clients filtered! Falling back to simple mean.")
            survivors = list(range(n_participants))
        
        survivors_updates = updates_matrix[survivors]
        survivors_ids = [client_ids[i] for i in survivors]
        survivors_samples = [num_samples[i] for i in survivors]
        
        if self.verbose:
            print(f"After pre-screening: {len(survivors)}/{n_participants} survivors")
        
        # ========================================================================
        # STAGE 3: Coordinate Sanitization
        # ========================================================================
        sanitized_updates = self._coordinate_sanitization(survivors_updates)
        
        # ========================================================================
        # STAGE 4: Core Robust Estimators
        # ========================================================================
        estimators = self._compute_core_estimators(sanitized_updates, survivors_samples)
        
        # ========================================================================
        # STAGE 5: Ensemble Mixture via Hedge
        # ========================================================================
        ensemble_aggregate, hedge_metadata = self._ensemble_mixture(estimators)
        
        # ========================================================================
        # STAGE 6: Reputation Update
        # ========================================================================
        self._update_reputation(survivors_updates, ensemble_aggregate, survivors_ids)
        
        # ========================================================================
        # STAGE 7: Final Reputation-Weighted Aggregation
        # ========================================================================
        final_aggregate = self._reputation_weighted_aggregate(
            survivors_updates, survivors_ids, survivors_samples, ensemble_aggregate
        )
        
        # ========================================================================
        # STAGE 8: State Refresh
        # ========================================================================
        self._refresh_state(survivors_updates)
        
        # Update reputation statistics (store only summary, not full array to save memory)
        current_reputations = [self.reputation_scores_[cid] for cid in survivors_ids]
        self.statistics_['reputation_history'].append({
            'mean': float(np.mean(current_reputations)),
            'min': float(np.min(current_reputations)),
            'max': float(np.max(current_reputations)),
        })
        
        # Compile metadata for logging
        metadata = {
            'round': self.round_counter_,
            'n_participants': n_participants,
            'n_survivors': len(survivors),
            'filtered_indices': list(set(range(n_participants)) - set(survivors)),
            **filtered_metadata,
            **hedge_metadata,
            'avg_reputation': float(np.mean(current_reputations)),
            'min_reputation': float(np.min(current_reputations)),
        }
        
        return final_aggregate, metadata
    
    def _multi_view_prescreening(
        self, updates: NDArray, client_ids: List[int], num_samples: List[int]
    ) -> Tuple[List[int], Dict]:
        """
        Apply multiple complementary anomaly detection views.
        
        Returns indices of surviving clients and filtering metadata.
        """
        n = len(updates)
        survivors = set(range(n))
        metadata = {}
        
        # View 1: Norm-based filtering (catches gradient explosion)
        norm_survivors = self._filter_by_norm(updates)
        filtered_by_norm = survivors - norm_survivors
        survivors &= norm_survivors
        metadata['n_filtered_norm'] = len(filtered_by_norm)
        self.statistics_['filtered_by_norm'].append(len(filtered_by_norm))
        
        if self.verbose:
            print(f"  [Norm Filter] Removed {len(filtered_by_norm)} clients")
        
        # View 2: Cosine similarity filtering (catches direction attacks)
        if len(survivors) > 2:
            cosine_survivors = self._filter_by_cosine(updates, list(survivors))
            filtered_by_cosine = survivors - cosine_survivors
            survivors &= cosine_survivors
            metadata['n_filtered_cosine'] = len(filtered_by_cosine)
            self.statistics_['filtered_by_cosine'].append(len(filtered_by_cosine))
            
            if self.verbose:
                print(f"  [Cosine Filter] Removed {len(filtered_by_cosine)} clients")
        else:
            metadata['n_filtered_cosine'] = 0
        
        # View 3: Mahalanobis distance filtering (catches statistical outliers)
        if self.enable_mahalanobis and len(survivors) > 5 and self.robust_covariance_ is not None:
            maha_survivors = self._filter_by_mahalanobis(updates, list(survivors))
            filtered_by_maha = survivors - maha_survivors
            survivors &= maha_survivors
            metadata['n_filtered_mahalanobis'] = len(filtered_by_maha)
            self.statistics_['filtered_by_mahalanobis'].append(len(filtered_by_maha))
            
            if self.verbose:
                print(f"  [Mahalanobis Filter] Removed {len(filtered_by_maha)} clients")
        else:
            metadata['n_filtered_mahalanobis'] = 0
        
        # View 4: Clustering-based consensus (catches coordinated attacks)
        if self.enable_clustering and len(survivors) > 5:
            cluster_survivors = self._filter_by_clustering(updates, list(survivors))
            filtered_by_cluster = survivors - cluster_survivors
            survivors &= cluster_survivors
            metadata['n_filtered_clustering'] = len(filtered_by_cluster)
            self.statistics_['filtered_by_clustering'].append(len(filtered_by_cluster))
            
            if self.verbose:
                print(f"  [Clustering Filter] Removed {len(filtered_by_cluster)} clients")
        else:
            metadata['n_filtered_clustering'] = 0
        
        return list(survivors), metadata
    
    def _filter_by_norm(self, updates: NDArray) -> set:
        """Filter updates with abnormal L2 norms."""
        norms = np.linalg.norm(updates, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Adaptive threshold: mean ± k*std
        upper_bound = mean_norm + self.norm_clip_scale * std_norm
        lower_bound = max(0, mean_norm - self.norm_clip_scale * std_norm)
        
        survivors = set(np.where((norms >= lower_bound) & (norms <= upper_bound))[0])
        return survivors
    
    def _filter_by_cosine(self, updates: NDArray, candidates: List[int]) -> set:
        """Filter updates with low cosine similarity to median direction."""
        if len(candidates) < 2:
            return set(candidates)
        
        # Compute median direction as reference
        median_direction = np.median(updates[candidates], axis=0)
        median_norm = np.linalg.norm(median_direction)
        
        if median_norm < 1e-10:
            return set(candidates)  # Can't compute cosine with zero vector
        
        median_direction = median_direction / median_norm
        
        # Compute cosine similarity for each candidate
        similarities = []
        for idx in candidates:
            update = updates[idx]
            norm = np.linalg.norm(update)
            if norm < 1e-10:
                similarities.append(0.0)
            else:
                cosine_sim = np.dot(update / norm, median_direction)
                similarities.append(cosine_sim)
        
        # Adaptive threshold if not specified
        if self.cosine_threshold is None:
            threshold = np.median(similarities) - 2 * np.std(similarities)
        else:
            threshold = self.cosine_threshold
        
        survivors = set([candidates[i] for i, sim in enumerate(similarities) if sim >= threshold])
        return survivors
    
    def _filter_by_mahalanobis(self, updates: NDArray, candidates: List[int]) -> set:
        """Filter updates with high Mahalanobis distance (statistical outliers)."""
        if self.robust_covariance_ is None or len(candidates) < 5:
            return set(candidates)
        
        try:
            # Compute Mahalanobis distance for each candidate
            distances = self.robust_covariance_.mahalanobis(updates[candidates])
            
            # Threshold based on chi-squared distribution
            # For high-dim data, use adaptive threshold
            threshold = self.mahalanobis_threshold
            
            survivors = set([candidates[i] for i, d in enumerate(distances) if d <= threshold])
            return survivors if len(survivors) > 0 else set(candidates)
        except Exception as e:
            if self.verbose:
                print(f"  [Mahalanobis] Warning: {e}, skipping")
            return set(candidates)
    
    def _filter_by_clustering(self, updates: NDArray, candidates: List[int]) -> set:
        """Filter updates using DBSCAN clustering to find outlier noise points."""
        if len(candidates) < 5:
            return set(candidates)
        
        # Use PCA or random projection for dimensionality reduction if needed
        candidate_updates = updates[candidates]
        
        # Standardize before clustering
        scaler = StandardScaler()
        scaled_updates = scaler.fit_transform(candidate_updates)
        
        # DBSCAN to find core clusters and outliers (label = -1)
        # eps and min_samples are critical: tune based on expected malicious fraction
        eps = np.percentile(cdist(scaled_updates, scaled_updates, 'euclidean').flatten(), 20)
        min_samples = max(2, int(len(candidates) * (1 - self.malicious_fraction) * 0.5))
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(scaled_updates)
        
        # Keep only clients in core clusters (not noise points labeled -1)
        core_mask = labels != -1
        
        # Additional check: if too many are marked as noise, keep them (might be legitimate diversity)
        noise_fraction = np.sum(~core_mask) / len(candidates)
        if noise_fraction > self.malicious_fraction * 1.5:
            # Too aggressive, keep all
            return set(candidates)
        
        survivors = set([candidates[i] for i in range(len(candidates)) if core_mask[i]])
        return survivors if len(survivors) > 0 else set(candidates)
    
    def _coordinate_sanitization(self, updates: NDArray) -> NDArray:
        """
        Apply coordinate-wise trimming and winsorization to neutralize sparse spikes.
        
        This prevents attacks that manipulate specific coordinates while keeping
        overall statistics (norm, direction) relatively normal.
        """
        n, d = updates.shape
        sanitized = np.copy(updates)
        
        # Coordinate-wise winsorization: clip extreme values per dimension
        trim_count = max(1, int(n * self.trim_ratio))
        
        for j in range(d):
            coord_values = sanitized[:, j]
            # Compute trimmed bounds
            sorted_vals = np.sort(coord_values)
            lower_bound = sorted_vals[trim_count]
            upper_bound = sorted_vals[-trim_count - 1]
            
            # Winsorize: clip to bounds instead of removing
            sanitized[:, j] = np.clip(coord_values, lower_bound, upper_bound)
        
        return sanitized
    
    def _compute_core_estimators(
        self, updates: NDArray, num_samples: List[int]
    ) -> Dict[str, NDArray]:
        """
        Compute 5 core robust estimators with complementary strengths.
        
        1. Trimmed-Mean: Robust to outliers in magnitude
        2. Coordinate-wise Median: Robust to direction attacks
        3. Geometric Median: Robust in high dimensions
        4. Multi-Krum: Robust to colluding minorities
        5. AGR-style rescaling: Adaptive to attack strength
        """
        estimators = {}
        
        # Estimator 1: Trimmed-Mean
        estimators['trimmed_mean'] = self._trimmed_mean(updates, num_samples)
        
        # Estimator 2: Coordinate-wise Median
        estimators['median'] = self._coordinate_median(updates)
        
        # Estimator 3: Geometric Median (L1 in parameter space)
        estimators['geometric_median'] = self._geometric_median(updates, num_samples)
        
        # Estimator 4: Multi-Krum
        estimators['multi_krum'] = self._multi_krum(updates)
        
        # Estimator 5: AGR-style (Adaptive Gradient Rescaling)
        estimators['agr'] = self._agr_rescaling(updates, num_samples)
        
        return estimators
    
    def _trimmed_mean(self, updates: NDArray, num_samples: List[int]) -> NDArray:
        """Compute weighted trimmed mean (trim from both tails)."""
        # For simplicity, use unweighted trimming
        # Can extend to weighted version
        trim_proportion = self.trim_ratio
        return trim_mean(updates, trim_proportion, axis=0)
    
    def _coordinate_median(self, updates: NDArray) -> NDArray:
        """Coordinate-wise median."""
        return np.median(updates, axis=0)
    
    def _geometric_median(
        self, updates: NDArray, num_samples: List[int], max_iter: int = 80, tol: float = 1e-5
    ) -> NDArray:
        """
        Compute geometric median using Weiszfeld's algorithm.
        
        The geometric median minimizes sum of Euclidean distances, making it
        robust to outliers in all directions.
        """
        # Initialize with coordinate-wise median
        median = np.median(updates, axis=0)
        
        for iteration in range(max_iter):
            # Compute distances from current estimate
            distances = np.linalg.norm(updates - median, axis=1)
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Weighted average with inverse distance weights
            weights = 1.0 / distances
            weights /= np.sum(weights)
            
            new_median = np.sum(updates * weights[:, np.newaxis], axis=0)
            
            # Check convergence
            if np.linalg.norm(new_median - median) < tol:
                break
            
            median = new_median
        
        return median
    
    def _multi_krum(self, updates: NDArray, f: Optional[int] = None) -> NDArray:
        """
        Multi-Krum: Select m clients with smallest sum of distances to nearest neighbors.
        
        Parameters
        ----------
        f : int, optional
            Number of Byzantine clients to tolerate. If None, estimated from malicious_fraction.
        """
        n = len(updates)
        if f is None:
            f = int(n * self.malicious_fraction)
        
        m = n - f  # Number of clients to select
        m = max(1, min(m, n))
        
        # Compute pairwise distances
        distances = cdist(updates, updates, 'euclidean')
        
        # For each client, compute sum of distances to k nearest neighbors
        # k = n - f - 2 (exclude self and f Byzantine)
        k = max(1, n - f - 2)
        
        scores = []
        for i in range(n):
            # Sort distances for client i (exclude distance to self at index 0)
            sorted_distances = np.sort(distances[i])
            # Sum of k smallest distances (excluding self)
            score = np.sum(sorted_distances[1:k+1])
            scores.append(score)
        
        # Select m clients with smallest scores
        selected_indices = np.argsort(scores)[:m]
        
        # Return average of selected clients
        return np.mean(updates[selected_indices], axis=0)
    
    def _agr_rescaling(self, updates: NDArray, num_samples: List[int]) -> NDArray:
        """
        AGR (Adaptive Gradient Rescaling): Detect and rescale suspicious updates.
        
        Idea: If an update is too far from the mean, rescale it to be within
        a trust radius. This prevents large-magnitude attacks while preserving direction.
        """
        # Compute mean as reference
        mean_update = np.mean(updates, axis=0)
        
        rescaled_updates = []
        for update in updates:
            distance = np.linalg.norm(update - mean_update)
            
            # Compute trust radius as median distance * scale factor
            all_distances = np.linalg.norm(updates - mean_update, axis=1)
            trust_radius = np.median(all_distances) * 2.0
            
            if distance > trust_radius:
                # Rescale to trust radius boundary
                direction = (update - mean_update) / (distance + 1e-10)
                rescaled = mean_update + direction * trust_radius
                rescaled_updates.append(rescaled)
            else:
                rescaled_updates.append(update)
        
        return np.mean(rescaled_updates, axis=0)
    
    def _ensemble_mixture(self, estimators: Dict[str, NDArray]) -> Tuple[NDArray, Dict]:
        """
        Combine estimators using Hedge algorithm with online learning.
        
        If validation_loss_fn is provided, update weights based on performance.
        Otherwise, use current weights.
        """
        estimator_names = ['trimmed_mean', 'median', 'geometric_median', 'multi_krum', 'agr']
        estimator_values = np.array([estimators[name] for name in estimator_names])
        
        # Compute mixture
        mixture = np.sum(estimator_values * self.ensemble_weights_[:, np.newaxis], axis=0)
        
        metadata = {
            'ensemble_weights': self.ensemble_weights_.copy(),
        }
        
        # Update weights if validation function available
        if self.validation_loss_fn is not None:
            losses = []
            for est_value in estimator_values:
                loss = self.validation_loss_fn(est_value)
                losses.append(loss)
            
            losses = np.array(losses)
            metadata['estimator_losses'] = losses
            self.statistics_['ensemble_losses'].append(losses)
            
            # Hedge update: w_t+1 = w_t * exp(-lr * loss_t)
            self.ensemble_weights_ *= np.exp(-self.learning_rate * losses)
            self.ensemble_weights_ /= np.sum(self.ensemble_weights_)  # Normalize
            
            # Track which estimator had lowest loss
            best_estimator = estimator_names[np.argmin(losses)]
            self.statistics_['selected_estimator'].append(best_estimator)
            metadata['best_estimator'] = best_estimator
        
        return mixture, metadata
    
    def _update_reputation(
        self, updates: NDArray, ensemble_aggregate: NDArray, client_ids: List[int]
    ):
        """
        Update reputation scores based on conformity to the ensemble aggregate.
        
        Conformity metric: How close is the client's update to the robust aggregate?
        Higher conformity → higher reputation
        """
        for i, cid in enumerate(client_ids):
            # Compute conformity as inverse distance (normalized)
            distance = np.linalg.norm(updates[i] - ensemble_aggregate)
            
            # All distances for normalization
            all_distances = np.linalg.norm(updates - ensemble_aggregate, axis=1)
            median_distance = np.median(all_distances)
            
            # Conformity score: 1 if at median, >1 if closer, <1 if farther
            if median_distance > 1e-10:
                conformity = median_distance / (distance + 1e-10)
            else:
                conformity = 1.0
            
            # Clip conformity to [0, 2] and convert to [0, 1]
            conformity = np.clip(conformity, 0, 2) / 2.0
            
            # Exponential moving average update
            old_reputation = self.reputation_scores_[cid]
            new_reputation = self.reputation_decay * old_reputation + (1 - self.reputation_decay) * conformity
            self.reputation_scores_[cid] = np.clip(new_reputation, 0, 1)
        
        # Reputation history is now stored as summary in aggregate() method - removed to save memory
    
    def _reputation_weighted_aggregate(
        self,
        updates: NDArray,
        client_ids: List[int],
        num_samples: List[int],
        ensemble_aggregate: NDArray,
    ) -> NDArray:
        """
        Final aggregation weighted by reputation and sample counts.
        
        Combines:
        - Reputation scores (higher reputation → higher weight)
        - Sample counts (more data → higher weight)
        - Ensemble aggregate as anchor
        """
        # Get reputation scores for participants
        reputations = np.array([self.reputation_scores_[cid] for cid in client_ids])
        samples = np.array(num_samples)
        
        # Combined weight: reputation * samples
        weights = reputations * samples
        weights /= np.sum(weights)
        
        # Weighted average
        reputation_aggregate = np.sum(updates * weights[:, np.newaxis], axis=0)
        
        # Blend with ensemble aggregate (90% reputation-weighted, 10% ensemble)
        # This provides stability
        final_aggregate = 0.9 * reputation_aggregate + 0.1 * ensemble_aggregate
        
        return final_aggregate
    
    def _refresh_state(self, updates: NDArray):
        """
        Update internal state for next round.
        
        - Add recent updates to historical buffer (for covariance estimation)
        - Re-estimate robust covariance periodically
        """
        # Add to history (keep last 20 updates for efficiency - REDUCED for memory)
        # With 25M parameters, even 20 updates = 2GB, but better than 100 updates = 10GB
        if len(updates) > 0:
            self.historical_updates_.extend(updates)
            if len(self.historical_updates_) > 20:  # REDUCED from 100
                self.historical_updates_ = self.historical_updates_[-20:]
        
        # Re-estimate robust covariance every 5 rounds (expensive operation)
        if self.enable_mahalanobis and self.round_counter_ % 5 == 0 and len(self.historical_updates_) >= 10:
            try:
                # Use Minimum Covariance Determinant (MCD) for robust estimation
                # Note: For very high-dim data, consider PCA preprocessing
                self.robust_covariance_ = MinCovDet(random_state=42).fit(
                    np.array(self.historical_updates_)
                )
            except Exception as e:
                if self.verbose:
                    print(f"  [State Refresh] Covariance estimation failed: {e}")
    
    def get_statistics(self) -> Dict:
        """Return collected statistics for analysis."""
        return self.statistics_.copy()
    
    def reset_statistics(self):
        """Clear statistics (useful for new experiment runs)."""
        for key in self.statistics_:
            self.statistics_[key] = []
