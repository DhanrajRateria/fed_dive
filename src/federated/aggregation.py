"""
Aggregation strategies for federated learning.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

class AggregationStrategy(ABC):
    """Base abstract class for aggregation strategies."""

    @abstractmethod
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AggregationStrategy':
        """Create strategy instance from configuration."""
        strategy_type = config.get('name', 'fedavg').lower()
        logger.info(f"Creating aggregation strategy of type: {strategy_type}")

        if strategy_type == 'fedavg':
            return FedAvg()
        # FedProx config is handled during client training, server aggregation is FedAvg
        # We might still want to pass mu for logging/config consistency, but it's not used here.
        elif strategy_type == 'fedprox':
            mu = float(config.get('mu', 0.01))  # Ensure mu is a float
            logger.info(f"Using FedProx aggregation (server-side is FedAvg). Mu={mu} (applied client-side).")
            return FedProx(mu=mu)  # Aggregation is FedAvg
        elif strategy_type == 'median':
            return Median()
        elif strategy_type == 'trimmed_mean':
            trim_ratio = float(config.get('trim_ratio', 0.1))  # Ensure trim_ratio is a float
            logger.info(f"TrimmedMean trim_ratio: {trim_ratio}")
            return TrimmedMean(trim_ratio=trim_ratio)
        elif strategy_type == 'feddive':
            momentum = float(config.get('momentum', 0.9))  # Ensure momentum is a float
            epsilon = float(config.get('epsilon', 1e-8))  # Ensure epsilon is a float
            temperature = float(config.get('temperature', 1.0))  # Ensure temperature is a float
            normalize_distances = bool(config.get('normalize_distances', True))  # Ensure boolean
            logger.info(f"FedDive parameters: momentum={momentum}, epsilon={epsilon}, "
                        f"temperature={temperature}, normalize_distances={normalize_distances}")
            return FedDive(momentum=momentum, epsilon=epsilon, temperature=temperature,
                           normalize_distances=normalize_distances)
        elif strategy_type == 'feddiver':
            momentum = float(config.get('momentum', 0.9))  # Ensure momentum is a float
            epsilon = float(config.get('epsilon', 1e-8))  # Ensure epsilon is a float
            temperature = float(config.get('temperature', 1.0))  # Ensure temperature is a float
            normalize_distances = bool(config.get('normalize_distances', True))  # Ensure boolean
            logger.info(f"FedDive-R parameters: momentum={momentum}, epsilon={epsilon}, "
                        f"temperature={temperature}, normalize_distances={normalize_distances}")
            return FedDiveR(momentum=momentum, epsilon=epsilon, temperature=temperature,
                            normalize_distances=normalize_distances)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy."""

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using weighted averaging."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("FedAvg: No client updates provided for aggregation.")
            return {}

        logger.info(f"FedAvg: Aggregating updates from {num_clients} clients.")

        # Extract parameters (state dicts) and weights
        # Assume parameters are already on CPU from client.get_parameters()
        parameters_list = [params for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates], dtype=np.float32) # Use float32 for weights

        logger.debug(f"FedAvg: Client weights (sample counts): {weights}")

        # Check for invalid weights
        if np.any(weights < 0):
            logger.error("FedAvg: Received negative weights, aborting aggregation.")
            raise ValueError("Client weights cannot be negative.")

        # Normalize weights
        total_weight = weights.sum()
        if total_weight <= 0:
            logger.warning(f"FedAvg: Total weight is {total_weight}. Using equal weights for {num_clients} clients.")
            normalized_weights = np.ones(num_clients, dtype=np.float32) / num_clients
        else:
            normalized_weights = weights / total_weight
            logger.debug(f"FedAvg: Normalized weights: {normalized_weights}")


        # Initialize aggregated result (use structure of the first client's update)
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("FedAvg: First client update dictionary is empty.")
             return {}

        for name, tensor in first_client_params.items():
            aggregated_params[name] = torch.zeros_like(tensor, dtype=torch.float32) # Ensure float32 for accumulation

        # Perform weighted averaging
        for client_idx, client_params in enumerate(parameters_list):
            client_weight = normalized_weights[client_idx]
            if not client_params:
                 logger.warning(f"FedAvg: Client update at index {client_idx} is empty, skipping.")
                 continue

            for name, param_tensor in client_params.items():
                if name in aggregated_params:
                    # Ensure tensors are float for accumulation if they aren't already
                    aggregated_params[name] += client_weight * param_tensor.float()
                else:
                     logger.warning(f"FedAvg: Parameter '{name}' from client {client_idx} "
                                    f"not found in the first client's structure. Skipping.")

        # Optional: Convert back to original dtype if necessary, though float32 is common
        # for name in aggregated_params:
        #     aggregated_params[name] = aggregated_params[name].to(first_client_params[name].dtype)

        logger.info(f"FedAvg: Aggregation complete.")
        return aggregated_params


# FedProx on server just uses FedAvg aggregation
class FedProx(FedAvg):
     """Server-side aggregation for FedProx (identical to FedAvg)."""
     def __init__(self, mu: float = 0.0): # Mu is unused server-side but kept for consistency
        super().__init__()
        self.mu = float(mu) # Ensure mu is a float
        logger.debug(f"FedProx aggregation strategy initialized server-side (uses FedAvg). Mu={self.mu} (applied client-side).")

     # aggregate method is inherited from FedAvg


class Median(AggregationStrategy):
    """Coordinate-wise median aggregation strategy."""

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using coordinate-wise median."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("Median: No client updates provided.")
            return {}

        logger.info(f"Median: Aggregating updates from {num_clients} clients.")
        parameters_list = [params for params, _ in client_updates]

        # Initialize result
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("Median: First client update dictionary is empty.")
             return {}

        # Iterate through parameter names from the first client
        for name in first_client_params.keys():
            try:
                # Stack corresponding tensors from all clients (ensure float for median)
                # Handle cases where a client might not have the param (shouldn't happen with state_dict)
                tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
                if len(tensors_to_stack) != num_clients:
                     logger.warning(f"Median: Parameter '{name}' missing from some clients. "
                                    f"Aggregating over {len(tensors_to_stack)} values.")
                if not tensors_to_stack:
                     logger.warning(f"Median: No tensors found for parameter '{name}'. Skipping.")
                     continue

                stacked_tensors = torch.stack(tensors_to_stack, dim=0)
                # Compute median along the client dimension (dim=0)
                median_values = torch.median(stacked_tensors, dim=0).values
                # Convert back to original dtype if needed
                aggregated_params[name] = median_values.to(first_client_params[name].dtype)
                logger.debug(f"Median: Aggregated parameter '{name}' shape: {aggregated_params[name].shape}")
            except KeyError:
                 logger.warning(f"Median: Parameter '{name}' from first client not found in subsequent client, skipping.")
            except Exception as e:
                 logger.error(f"Median: Error aggregating parameter '{name}': {e}", exc_info=True)
                 # Decide whether to skip this param or raise error

        logger.info("Median: Aggregation complete.")
        return aggregated_params


class TrimmedMean(AggregationStrategy):
    """Trimmed mean aggregation strategy."""

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__()
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be between 0 (inclusive) and 0.5 (exclusive)")
        self.trim_ratio = float(trim_ratio)  # Ensure trim_ratio is a float
        logger.debug(f"TrimmedMean strategy initialized with trim_ratio={self.trim_ratio}")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using trimmed mean."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("TrimmedMean: No client updates provided.")
            return {}

        logger.info(f"TrimmedMean: Aggregating updates from {num_clients} clients (trim_ratio={self.trim_ratio}).")
        parameters_list = [params for params, _ in client_updates]

        # Calculate how many clients to trim from each end
        k = int(np.floor(self.trim_ratio * num_clients)) # Use floor to be safe
        logger.debug(f"TrimmedMean: Trimming k={k} clients from each end.")

        if 2 * k >= num_clients:
             logger.warning(f"TrimmedMean: Trim ratio ({self.trim_ratio}) too high for {num_clients} clients (k={k}). "
                            "Falling back to Median aggregation.")
             # Fallback to Median
             median_agg = Median()
             return median_agg.aggregate(client_updates)

        # Initialize result
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("TrimmedMean: First client update dictionary is empty.")
             return {}

        # For each parameter tensor name
        for name in first_client_params.keys():
            try:
                 # Stack corresponding tensors (ensure float for sorting/mean)
                 tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
                 if len(tensors_to_stack) != num_clients:
                     logger.warning(f"TrimmedMean: Parameter '{name}' missing from some clients. "
                                    f"Aggregating over {len(tensors_to_stack)} values.")
                     # TODO: Decide how to handle missing params - skip? error? adjust k?
                     # For now, we proceed but the trim might be less effective
                 if not tensors_to_stack:
                     logger.warning(f"TrimmedMean: No tensors found for parameter '{name}'. Skipping.")
                     continue

                 stacked_tensors = torch.stack(tensors_to_stack, dim=0)

                 # Sort along the client dimension (dim=0)
                 sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)

                 # Select the tensors after trimming k from both ends
                 trimmed_tensors = sorted_tensors[k : num_clients - k]

                 # Average the remaining tensors
                 mean_values = torch.mean(trimmed_tensors, dim=0)
                 aggregated_params[name] = mean_values.to(first_client_params[name].dtype) # Convert back
                 logger.debug(f"TrimmedMean: Aggregated parameter '{name}' shape: {aggregated_params[name].shape} "
                              f"(trimmed {2*k}/{num_clients} values)")
            except KeyError:
                 logger.warning(f"TrimmedMean: Parameter '{name}' from first client not found in subsequent client, skipping.")
            except Exception as e:
                 logger.error(f"TrimmedMean: Error aggregating parameter '{name}': {e}", exc_info=True)

        logger.info("TrimmedMean: Aggregation complete.")
        return aggregated_params


class FedDive(AggregationStrategy):
    """
    Implementation of the Federated Diversity Averaging (FedDive) algorithm.
    
    Reference (Algorithm 1 FEDDIVE): 
    1) Maintains a momentum-based velocity.
    2) Computes distances of client updates from the expected consensus position (previous params + velocity).
    3) Uses temperature-scaled softmax weighting to produce diversity-based aggregation.
    
    Args:
        momentum (float): Momentum coefficient (0 <= momentum < 1).
        epsilon (float): Small constant for numerical stability (e.g. 1e-8).
        temperature (float): Temperature parameter for softmax weighting.
        normalize_distances (bool): If True, normalizes distances (mean 0, std 1) before softmax.
    """
    def __init__(
        self,
        momentum: float = 0.9,
        epsilon: float = 1e-8,
        temperature: float = 1.0,
        normalize_distances: bool = True
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be in [0,1).")
        self.momentum = float(momentum)  # Ensure float
        self.epsilon = float(epsilon)    # Ensure float
        self.temperature = float(temperature)  # Ensure float
        self.normalize_distances = bool(normalize_distances)  # Ensure boolean

        # Internal states that persist across aggregation rounds:
        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}

    def _init_state(self, reference_params: Dict[str, torch.Tensor]):
        """
        Initialize velocity and previous_global_params using the reference (first round).
        """
        self.velocity = {
            name: torch.zeros_like(param) for name, param in reference_params.items()
        }
        self.previous_global_params = {
            name: param.clone() for name, param in reference_params.items()
        }

    def _calc_velocity(self, new_global_params: Dict[str, torch.Tensor]):
        """
        Update the velocity using the difference (deltas) between new_global_params and previous_global_params
        in a momentum fashion.
        """
        # Compute deltas
        deltas = {}
        for name, param in new_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            if prev_param is not None:
                deltas[name] = param - prev_param
            else:
                # If it doesn't exist, treat the entire param as delta
                deltas[name] = param

        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    + (1.0 - self.momentum) * deltas[name]
                )

    def _calc_distances(
        self, client_params_list: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """
        Calculate L2 distances of each client update from the expected consensus position,
        defined as (previous_global_params + velocity).
        
        Returns a numpy array of distances (floating-point).
        """
        # Expected consensus position
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel

        distances = []
        for client_params in client_params_list:
            dist_sq = 0.0
            for name, expected_val in expected_position.items():
                client_val = client_params.get(name, None)
                if client_val is not None:
                    diff = client_val - expected_val
                    dist_sq += torch.sum(diff * diff).item()
                else:
                    # If client doesn't have this parameter, treat it as zero in difference
                    dist_sq += torch.sum(expected_val * expected_val).item()
            distances.append(np.sqrt(dist_sq))
        return np.array(distances)

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Main aggregation routine.
        
        Args:
            client_updates: A list of (parameter_dict, sample_count). The sample_count is not
                            explicitly used in FedDive's weighting, but is retained for compatibility.
        
        Returns:
            A dictionary representing the new globally aggregated parameters.
        """
        if not client_updates:
            logger.warning("[FedDive] No client updates received. Returning empty aggregation.")
            return {}

        # Extract client parameter dicts
        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDive] The first client update is empty. Returning empty dict.")
            return {}

        num_clients = len(params_list)
        logger.info(f"[FedDive] Aggregating updates from {num_clients} clients.")

        # Step 1: Simple average for a baseline global estimate
        new_global_params = {}
        for name, tensor in params_list[0].items():
            new_global_params[name] = torch.zeros_like(tensor)

        for client_params in params_list:
            for name, tensor in client_params.items():
                if name not in new_global_params:
                    new_global_params[name] = torch.zeros_like(tensor)
                new_global_params[name] += tensor
        for name in new_global_params:
            new_global_params[name] /= float(num_clients)

        # If first round or mismatch in state, initialize
        if not self.velocity or not self.previous_global_params:
            self._init_state(new_global_params)
            logger.debug("[FedDive] Initialized state vectors.")

        # Step 2: Update velocity
        self._calc_velocity(new_global_params)
        logger.debug("[FedDive] Updated velocity.")

        # Step 3: Calculate distances from the expected consensus
        distances = self._calc_distances(params_list)
        logger.debug(f"[FedDive] Client distances: {distances}")

        # Step 4: (Optional) Distance normalization
        if self.normalize_distances and len(distances) > 1:
            mean_val = float(distances.mean())
            std_val = float(distances.std())  # Ensure std_val is a float
            if std_val < self.epsilon:  # Now comparing float to float
                std_val = 1.0
            distances = (distances - mean_val) / std_val
            logger.debug(f"[FedDive] Normalized distances: {distances}")

        # Step 5: Temperature-scaled softmax for weighting
        scaled = distances / float(max(self.temperature, self.epsilon))  # Ensure float division
        # Add an epsilon to the exponent to avoid large negative exponents
        exp_vals = np.exp(scaled + self.epsilon)
        diversity_weights = exp_vals / np.sum(exp_vals)
        logger.debug(f"[FedDive] Diversity weights: {diversity_weights}")

        # Step 6: Weighted aggregation of client params
        aggregated_params = {
            name: torch.zeros_like(tensor) for name, tensor in params_list[0].items()
        }
        for idx, client_params in enumerate(params_list):
            w = diversity_weights[idx]
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        # Update previous_global_params for the next round
        self.previous_global_params = {
            name: p.clone() for name, p in aggregated_params.items()
        }

        logger.info("[FedDive] Aggregation complete.")
        return aggregated_params


class FedDiveR(AggregationStrategy):
    """
    FEDDIVE-R: Robust Federated Diversity Exploration.
    
    This variant (Algorithm 2) uses a 'robust consensus tracking' step:
    - Instead of computing the average client update for momentum, it uses the median 
      across all client updates as the robust consensus.
    - The rest of the steps (distance measurement, softmax weighting) remain the same 
      as in FedDive.
    
    Args:
        momentum (float): Momentum coefficient (0 <= momentum < 1).
        epsilon (float): Small constant for numerical stability (e.g. 1e-8).
        temperature (float): Temperature parameter for softmax weighting.
        normalize_distances (bool): If True, normalizes distances before applying softmax.
    """
    def __init__(
        self,
        momentum: float = 0.9,
        epsilon: float = 1e-8,
        temperature: float = 1.0,
        normalize_distances: bool = True
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be in [0,1).")
        self.momentum = float(momentum)  # Ensure float
        self.epsilon = float(epsilon)    # Ensure float
        self.temperature = float(temperature)  # Ensure float
        self.normalize_distances = bool(normalize_distances)  # Ensure boolean

        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}

    def _init_state(self, reference_params: Dict[str, torch.Tensor]):
        """
        Initialize velocity and previous_global_params if needed.
        """
        self.velocity = {
            name: torch.zeros_like(param) for name, param in reference_params.items()
        }
        self.previous_global_params = {
            name: param.clone() for name, param in reference_params.items()
        }

    def _calc_median_params(
        self, client_params_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the parameter-wise median across the list of client updates.
        Returns a dictionary of the same structure (keys) with median values.
        """
        # Collect parameter names
        param_names = list(client_params_list[0].keys())
        median_params = {}
        for name in param_names:
            # Gather all param[name] from each client
            stack = []
            for cp in client_params_list:
                if name in cp:
                    stack.append(cp[name].unsqueeze(0))
            if not stack:
                continue
            stacked_vals = torch.cat(stack, dim=0)
            # Flatten across clients; dimension is [num_clients, *param_shape]
            # We compute median along dimension 0
            median_tensor = torch.median(stacked_vals, dim=0)[0]
            median_params[name] = median_tensor
        return median_params

    def _calc_velocity(self, robust_global_params: Dict[str, torch.Tensor]):
        # Compute deltas from robust global to previous global
        deltas = {}
        for name, param in robust_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            if prev_param is not None:
                deltas[name] = param - prev_param
            else:
                deltas[name] = param

        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    + (1.0 - self.momentum) * deltas[name]
                )

    def _calc_distances(
        self, client_params_list: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        # Expected position
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel

        distances = []
        for client_params in client_params_list:
            dist_sq = 0.0
            for name, expected_val in expected_position.items():
                client_val = client_params.get(name, None)
                if client_val is not None:
                    diff = client_val - expected_val
                    dist_sq += torch.sum(diff * diff).item()
                else:
                    dist_sq += torch.sum(expected_val * expected_val).item()
            distances.append(np.sqrt(dist_sq))
        return np.array(distances)

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Main aggregation routine for FedDiveR (Algorithm 2).
        
        Args:
            client_updates: A list of (parameters, sample_count). sample_count is not used 
                            in weighting but is retained for interface consistency.
        
        Returns:
            A dictionary representing the new robust, diversity-based aggregated parameters.
        """
        if not client_updates:
            logger.warning("[FedDiveR] No client updates received. Returning empty aggregation.")
            return {}

        # Extract client param dicts
        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDiveR] The first client update is empty. Returning empty dict.")
            return {}

        num_clients = len(params_list)
        logger.info(f"[FedDiveR] Aggregating updates from {num_clients} clients.")

        # 1) Compute robust consensus (median) for momentum
        robust_global_params = self._calc_median_params(params_list)
        logger.debug("[FedDiveR] Computed robust consensus using parameter-wise median.")

        # If first round or mismatch in state, initialize
        if not self.velocity or not self.previous_global_params:
            self._init_state(robust_global_params)
            logger.debug("[FedDiveR] Initialized state vectors.")

        # 2) Update velocity with robust delta
        self._calc_velocity(robust_global_params)
        logger.debug("[FedDiveR] Updated velocity.")

        # 3) Calculate distances from the expected consensus
        distances = self._calc_distances(params_list)
        logger.debug(f"[FedDiveR] Client distances: {distances}")

        # 4) (Optional) distance normalization
        if len(distances) > 1:
            median_dist = np.median(distances)
            # MAD: Median Absolute Deviation
            mad = np.median(np.abs(distances - median_dist))
            # Robust Z-score. Use a high threshold like 3.0 or 3.5
            # The 1.4826 is a scaling factor to make MAD comparable to std dev for normal distributions
            robust_z_scores = np.abs(0.6745 * (distances - median_dist) / (mad + self.epsilon))
            
            # Identify outliers: anything beyond a certain z-score
            is_outlier = robust_z_scores > 3.0 
            logger.info(f"[FedDiveR] Outlier detection: median_dist={median_dist:.4f}, mad={mad:.4f}")
            logger.info(f"[FedDiveR] Robust Z-scores: {robust_z_scores}")
            logger.info(f"[FedDiveR] Identified outliers (by index): {np.where(is_outlier)[0]}")
            
        if self.normalize_distances and len(distances) > 1:
            mean_val = float(distances.mean())
            std_val = float(distances.std())  # Ensure std_val is a float
            if std_val < self.epsilon:  # Now comparing float to float
                std_val = 1.0
            distances = (distances - mean_val) / std_val
            logger.debug(f"[FedDiveR] Normalized distances: {distances}")

        # 5) Temperature-scaled softmax weighting
        scaled = distances / float(max(self.temperature, self.epsilon))  # Ensure float division
        exp_vals = np.exp(scaled + self.epsilon)

        exp_vals[is_outlier] = 0.0

        if np.sum(exp_vals)  < self.epsilon:
            logger.warning("[FedDiveR] All clients were flagged as outliers or weights are zero. Falling back to median.")

        diversity_weights = exp_vals / np.sum(exp_vals)
        logger.debug(f"[FedDiveR] Diversity weights (outliers rejected): {diversity_weights}")

        # 6) Weighted aggregation
        aggregated_params = {}
        for name, tensor in params_list[0].items():
            aggregated_params[name] = torch.zeros_like(tensor)

        for idx, client_params in enumerate(params_list):
            w = diversity_weights[idx]
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        # Update previous_global_params with the new aggregated parameters
        self.previous_global_params = {
            name: p.clone() for name, p in aggregated_params.items()
        }

        logger.info("[FedDiveR] Aggregation complete.")
        return aggregated_params