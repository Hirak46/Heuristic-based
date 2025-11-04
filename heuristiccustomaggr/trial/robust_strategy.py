"""
Custom Flower Strategy with CERBERUS-AGG

Integrates CERBERUS-AGG into Flower's federated learning framework.
"""

from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from trial.cerberus_agg import CerberusAggregator


class CerberusFedStrategy(FedAvg):
    """
    Custom Flower Strategy that uses CERBERUS-AGG for Byzantine-robust aggregation.
    
    Extends FedAvg to replace the aggregation step with CERBERUS-AGG.
    
    Parameters
    ----------
    num_clients : int
        Total number of clients in federation
    malicious_fraction : float
        Expected fraction of malicious clients
    aggregator_params : dict, optional
        Additional parameters for CERBERUS-AGG
    **kwargs
        Arguments passed to parent FedAvg strategy
    """
    
    def __init__(
        self,
        num_clients: int = 100,
        malicious_fraction: float = 0.3,
        aggregator_params: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_clients = num_clients
        self.malicious_fraction = malicious_fraction
        
        # Initialize CERBERUS-AGG
        agg_params = aggregator_params or {}
        self.aggregator = CerberusAggregator(
            num_clients=num_clients,
            malicious_fraction=malicious_fraction,
            **agg_params
        )
        
        self.aggregation_metadata_history = []
        
        print(f"\n{'='*80}")
        print(f"CERBERUS-AGG Strategy Initialized")
        print(f"  Num clients: {num_clients}")
        print(f"  Expected malicious fraction: {malicious_fraction}")
        print(f"{'='*80}\n")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using robust aggregation.
        
        This method is called by Flower server after each training round.
        We override it to use our Byzantine-robust aggregators instead of
        simple weighted averaging.
        """
        if not results:
            return None, {}
        
        # Extract client updates and metadata
        client_updates = []
        client_ids = []
        num_samples = []
        
        for client_proxy, fit_res in results:
            # Convert parameters to numpy arrays
            update = parameters_to_ndarrays(fit_res.parameters)
            # Flatten all layers into single vector
            flattened = self._flatten_params(update)
            
            client_updates.append(flattened)
            
            # Extract client ID from metrics or use proxy cid
            cid = fit_res.metrics.get('client_id', hash(client_proxy.cid) % self.num_clients)
            client_ids.append(cid)
            
            num_samples.append(fit_res.num_examples)
        
        # Apply robust aggregation
        aggregated_flat, metadata = self.aggregator.aggregate(
            client_updates=client_updates,
            client_ids=client_ids,
            num_samples=num_samples,
        )
        
        # Store metadata for analysis
        metadata['round'] = server_round
        self.aggregation_metadata_history.append(metadata)
        
        # Reconstruct parameter structure
        # Use first client's structure as template
        template_params = parameters_to_ndarrays(results[0][1].parameters)
        aggregated_params = self._unflatten_params(aggregated_flat, template_params)
        
        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_params)
        
        # Prepare metrics for logging
        metrics_aggregated = {
            'aggregator': 'cerberus',
            'n_participants': len(results),
        }
        
        # Add aggregator-specific metrics
        if 'n_survivors' in metadata:
            metrics_aggregated['n_filtered'] = len(results) - metadata['n_survivors']
        
        if 'avg_reputation' in metadata:
            metrics_aggregated['avg_reputation'] = metadata['avg_reputation']
        
        return parameters_aggregated, metrics_aggregated
    
    def _flatten_params(self, params: List[np.ndarray]) -> np.ndarray:
        """Flatten list of parameter arrays into single vector."""
        return np.concatenate([arr.flatten() for arr in params])
    
    def _unflatten_params(
        self, flat_params: np.ndarray, template: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Reconstruct parameter structure from flat vector using template."""
        params = []
        offset = 0
        
        for template_arr in template:
            shape = template_arr.shape
            size = int(np.prod(shape))  # Convert to int for slicing
            
            # Extract slice and reshape
            param_arr = flat_params[offset:offset + size].reshape(shape)
            params.append(param_arr)
            
            offset += size
        
        return params
    
    def get_aggregation_history(self) -> List[Dict]:
        """Return history of aggregation metadata for analysis."""
        return self.aggregation_metadata_history
