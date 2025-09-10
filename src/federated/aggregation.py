"""
Aggregation strategies for federated learning.

This module provides various aggregation strategies for federated learning scenarios,
including standard algorithms like FedAvg, Median, and robust approaches like 
TrimmedMean and FedDive variants.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    """Base abstract class for aggregation strategies.
    
    All concrete aggregation strategies should inherit from this class and
    implement the aggregate method.
    """

    @abstractmethod
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates.
        
        Args:
            client_updates: List of tuples containing client model parameters and 
                           their corresponding weights (often sample counts)
            current_round: Current training round (used for dynamic strategies)
            total_rounds: Total number of training rounds (used for dynamic strategies)
            
        Returns:
            Aggregated model parameters as a state dictionary
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AggregationStrategy':
        """Create strategy instance from configuration.
        
        Args:
            config: Dictionary containing strategy configuration parameters
            
        Returns:
            An initialized aggregation strategy instance
            
        Raises:
            ValueError: If an unsupported aggregation strategy is specified
        """
        strategy_type = config.get('name', 'fedavg').lower()
        logger.info(f"Creating aggregation strategy of type: {strategy_type}")

        if strategy_type == 'fedavg':
            return FedAvg()
        elif strategy_type == 'fedprox':
            mu = float(config.get('mu', 0.01))
            logger.info(f"Using FedProx aggregation (server-side is FedAvg). Mu={mu} (applied client-side).")
            return FedProx(mu=mu)
        elif strategy_type == 'median':
            return Median()
        elif strategy_type == 'trimmed_mean':
            trim_ratio = float(config.get('trim_ratio', 0.1))
            logger.info(f"TrimmedMean trim_ratio: {trim_ratio}")
            return TrimmedMean(trim_ratio=trim_ratio)
        elif strategy_type == 'feddive':
            # Fixed or dynamic
            common_kwargs = dict(
                momentum=float(config.get('momentum', 0.9)),
                epsilon=float(config.get('epsilon', 1e-8)),
                normalize_distances=bool(config.get('normalize_distances', True)),
                distance_metric=str(config.get('distance_metric', 'l2')).lower(),
                blend_samples=str(config.get('blend_samples', 'none')).lower(),
                warmup_rounds=int(config.get('warmup_rounds', 0))
            )
            if config.get('is_dynamic', False):
                logger.info("Initializing FedDive with DYNAMIC temperature scheduling.")
                return FedDive(
                    is_dynamic=True,
                    temp_schedule=config.get('temp_schedule', 'cosine'),
                    initial_temp=float(config.get('initial_temp', 5.0)),
                    final_temp=float(config.get('final_temp', 0.5)),
                    **common_kwargs
                )
            else:
                logger.info("Initializing FedDive with FIXED temperature.")
                return FedDive(
                    temperature=float(config.get('temperature', 1.0)),
                    **common_kwargs
                )
        elif strategy_type == 'feddiver':
            # Fixed or dynamic (robust)
            common_kwargs = dict(
                momentum=float(config.get('momentum', 0.9)),
                epsilon=float(config.get('epsilon', 1e-8)),
                normalize_distances=bool(config.get('normalize_distances', True)),
                distance_metric=str(config.get('distance_metric', 'l2')).lower(),
                blend_samples=str(config.get('blend_samples', 'none')).lower(),
                warmup_rounds=int(config.get('warmup_rounds', 0))
            )
            if config.get('is_dynamic', False):
                logger.info("Initializing FedDiveR with DYNAMIC temperature scheduling.")
                return FedDiveR(
                    is_dynamic=True,
                    temp_schedule=config.get('temp_schedule', 'cosine'),
                    initial_temp=float(config.get('initial_temp', 5.0)),
                    final_temp=float(config.get('final_temp', 0.5)),
                    **common_kwargs
                )
            else:
                logger.info("Initializing FedDiveR with FIXED temperature.")
                return FedDiveR(
                    temperature=float(config.get('temperature', 1.0)),
                    **common_kwargs
                )
        elif strategy_type == 'feddive_lw':
            logger.info("Initializing FedDive-LayerWise (FedDive-LW).")
            return FedDiveLW(
                divergence_layers=config.get('divergence_layers'),
                backbone_blend=str(config.get('backbone_blend', 'sqrt')).lower(),
                momentum=float(config.get('momentum', 0.9)),
                epsilon=float(config.get('epsilon', 1e-8)),
                temperature=float(config.get('temperature', 1.0)),
                normalize_distances=bool(config.get('normalize_distances', True)),
                distance_metric=str(config.get('distance_metric', 'l2')).lower(),
                blend_samples=str(config.get('blend_samples', 'none')).lower(),
                warmup_rounds=int(config.get('warmup_rounds', 0))
            )
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy.
    
    Implementation of the standard FedAvg algorithm (McMahan et al., 2017),
    which performs weighted averaging of client updates.
    """

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate using weighted averaging.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
            current_round: Current training round (ignored in FedAvg)
            total_rounds: Total number of rounds (ignored in FedAvg)
            
        Returns:
            Weighted averaged model parameters
        """
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("FedAvg: No client updates provided for aggregation.")
            return {}

        logger.info(f"FedAvg: Aggregating updates from {num_clients} clients.")

        # Extract parameters (state dicts) and weights
        parameters_list = [params for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates], dtype=np.float32)

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

        # Initialize aggregated result (use structure of the first client's update)
        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("FedAvg: First client update dictionary is empty.")
            return {}
             
        aggregated_params = {
            name: torch.zeros_like(tensor, dtype=torch.float32) 
            for name, tensor in first_client_params.items()
        }

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

        logger.info("FedAvg: Aggregation complete.")
        return aggregated_params


class FedProx(FedAvg):
    """Server-side aggregation for FedProx (identical to FedAvg).
    
    Note: FedProx differs from FedAvg in the client optimization process 
    (with proximal term), but server aggregation remains the same as FedAvg.
    """
    
    def __init__(self, mu: float = 0.01):
        """Initialize FedProx server aggregation.
        
        Args:
            mu: Proximal term weight (used client-side, not in server aggregation)
        """
        super().__init__()
        self.mu = float(mu)  # Ensure mu is a float
        logger.debug(f"FedProx aggregation strategy initialized server-side (uses FedAvg). "
                    f"Mu={self.mu} (applied client-side).")


class Median(AggregationStrategy):
    """Coordinate-wise median aggregation strategy.
    
    A robust aggregation method that computes element-wise median of client parameters.
    Provides robustness against certain Byzantine attacks.
    """

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate using coordinate-wise median.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
                          Note: sample_counts are ignored in Median aggregation
            current_round: Current training round (ignored in Median)
            total_rounds: Total number of rounds (ignored in Median)
            
        Returns:
            Model parameters aggregated using coordinate-wise median
        """
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("Median: No client updates provided.")
            return {}

        logger.info(f"Median: Aggregating updates from {num_clients} clients.")
        parameters_list = [params for params, _ in client_updates]

        # Initialize result
        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("Median: First client update dictionary is empty.")
            return {}

        aggregated_params = {}

        # Iterate through parameter names from the first client
        for name in first_client_params.keys():
            try:
                # Stack corresponding tensors from all clients (ensure float for median)
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
            except Exception as e:
                logger.error(f"Median: Error aggregating parameter '{name}': {e}", exc_info=True)

        logger.info("Median: Aggregation complete.")
        return aggregated_params


class TrimmedMean(AggregationStrategy):
    """Trimmed mean aggregation strategy.
    
    A robust aggregation method that removes the highest and lowest values
    before computing the mean. Provides robustness against Byzantine attacks.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """Initialize TrimmedMean aggregation.
        
        Args:
            trim_ratio: Fraction of values to trim from each end (between 0 and 0.5)
        
        Raises:
            ValueError: If trim_ratio is not in the valid range
        """
        super().__init__()
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be between 0 (inclusive) and 0.5 (exclusive)")
        self.trim_ratio = float(trim_ratio)
        logger.debug(f"TrimmedMean strategy initialized with trim_ratio={self.trim_ratio}")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate using trimmed mean.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
                          Note: sample_counts are ignored in TrimmedMean
            current_round: Current training round (ignored in TrimmedMean)
            total_rounds: Total number of rounds (ignored in TrimmedMean)
            
        Returns:
            Model parameters aggregated using trimmed mean
        """
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("TrimmedMean: No client updates provided.")
            return {}

        logger.info(f"TrimmedMean: Aggregating updates from {num_clients} clients (trim_ratio={self.trim_ratio}).")
        parameters_list = [params for params, _ in client_updates]

        # Calculate how many clients to trim from each end
        k = int(np.floor(self.trim_ratio * num_clients))
        logger.debug(f"TrimmedMean: Trimming k={k} clients from each end.")

        if 2 * k >= num_clients:
            logger.warning(f"TrimmedMean: Trim ratio ({self.trim_ratio}) too high for {num_clients} clients (k={k}). "
                          "Falling back to Median aggregation.")
            # Fallback to Median
            median_agg = Median()
            return median_agg.aggregate(client_updates, current_round, total_rounds)

        # Initialize result
        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("TrimmedMean: First client update dictionary is empty.")
            return {}

        aggregated_params = {}

        # For each parameter tensor name
        for name in first_client_params.keys():
            try:
                # Stack corresponding tensors (ensure float for sorting/mean)
                tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
                if len(tensors_to_stack) != num_clients:
                    logger.warning(f"TrimmedMean: Parameter '{name}' missing from some clients. "
                                  f"Aggregating over {len(tensors_to_stack)} values.")
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
                aggregated_params[name] = mean_values.to(first_client_params[name].dtype)
            except Exception as e:
                logger.error(f"TrimmedMean: Error aggregating parameter '{name}': {e}", exc_info=True)

        logger.info("TrimmedMean: Aggregation complete.")
        return aggregated_params


def _flatten_state_dict(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Utility function to flatten a state dict into a single tensor.
    
    Args:
        params: Dictionary of parameter tensors
        
    Returns:
        Flattened tensor containing all parameters
    """
    return torch.cat([p.detach().flatten().float() for p in params.values()])


class FedDive(AggregationStrategy):
    """Federated Diversity Exploration (FedDive) aggregation strategy.
    
    FedDive uses momentum-based velocity tracking and distance-based weighting to enhance
    model exploration during federated learning. It can use either fixed or dynamic temperature
    scheduling to control exploration-exploitation trade-offs during training.
    
    Features:
    - Distance metrics: 'l2' (default) or 'cosine' (directional divergence w.r.t velocity)
    - Sample blending: 'none' (default), 'sqrt', or 'linear' (post-softmax blend with sample fractions)
    - Dynamic temperature scheduling: cosine or linear annealing
    - Optional warmup rounds: use uniform or sample-based weights for initial rounds
    
    Reference: FedDive algorithm (Rateria et al., 2025)
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        epsilon: float = 1e-8,
        temperature: float = 1.0,
        normalize_distances: bool = True,
        is_dynamic: bool = False,
        temp_schedule: str = 'cosine',
        initial_temp: float = 5.0,
        final_temp: float = 0.5,
        distance_metric: str = 'l2',
        blend_samples: str = 'none',
        warmup_rounds: int = 0
    ):
        """Initialize FedDive aggregation strategy.
        
        Args:
            momentum: Momentum coefficient for velocity tracking (0 <= momentum < 1)
            epsilon: Small constant for numerical stability
            temperature: Temperature parameter for softmax weighting (fixed mode only)
            normalize_distances: Whether to normalize distances before softmax
            is_dynamic: Whether to use dynamic temperature scheduling
            temp_schedule: Schedule type for dynamic temperature ('cosine' or 'linear')
            initial_temp: Starting temperature for dynamic scheduling
            final_temp: Final temperature for dynamic scheduling
            distance_metric: Method to calculate distances ('l2' or 'cosine')
            blend_samples: How to blend diversity weights with sample counts ('none', 'sqrt', 'linear')
            warmup_rounds: Number of initial rounds to use uniform/sample-based weights
            
        Raises:
            ValueError: If momentum is not in [0,1)
        """
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be in [0,1).")
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.normalize_distances = bool(normalize_distances)
        
        # Dynamic temperature parameters
        self.is_dynamic = bool(is_dynamic)
        if self.is_dynamic:
            self.temp_schedule = temp_schedule
            self.initial_temp = float(initial_temp)
            self.final_temp = float(final_temp)
            self.temperature = self.initial_temp  # Start with initial temperature
            logger.info(f"FedDive Dynamic Temp enabled: schedule={temp_schedule}, "
                        f"start={initial_temp}, end={final_temp}")
        else:
            self.temperature = float(temperature)
            logger.info(f"FedDive Fixed Temp: {self.temperature}")

        # Distance metric and sample blending options
        self.distance_metric = distance_metric
        self.blend_samples = blend_samples
        self.warmup_rounds = int(warmup_rounds)
        
        logger.info(f"FedDive using distance metric '{distance_metric}', sample blend mode '{blend_samples}', "
                   f"and {warmup_rounds} warmup rounds")

        # Internal state variables that persist across rounds
        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}
        self._last_telemetry: Optional[Dict[str, float]] = None

    def _update_temperature(self, current_round: int, total_rounds: int):
        """Update temperature based on the training progress and selected schedule.
        
        Args:
            current_round: Current training round
            total_rounds: Total number of training rounds
        """
        if not self.is_dynamic or total_rounds <= 1:
            return

        progress = current_round / max(1, total_rounds - 1)  # Avoid division by zero
        
        if self.temp_schedule.lower() == 'cosine':
            # Cosine annealing schedule
            cosine_val = 0.5 * (1 + np.cos(np.pi * progress))
            self.temperature = self.final_temp + (self.initial_temp - self.final_temp) * cosine_val
        elif self.temp_schedule.lower() == 'linear':
            # Linear decay schedule
            self.temperature = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        else:
            raise ValueError(f"Unknown temperature schedule: {self.temp_schedule}")
        
        logger.debug(f"[FedDive] Round {current_round}/{total_rounds}, "
                     f"Dynamic Temp updated to: {self.temperature:.4f}")

    def _init_state(self, reference_params: Dict[str, torch.Tensor]):
        """Initialize velocity and previous_global_params using reference parameters.
        
        Args:
            reference_params: Model parameters to use as reference
        """
        self.velocity = {
            name: torch.zeros_like(param) for name, param in reference_params.items()
        }
        self.previous_global_params = {
            name: param.clone() for name, param in reference_params.items()
        }

    def _calc_velocity(self, new_global_params: Dict[str, torch.Tensor]):
        """Update the velocity using momentum and new parameters.
        
        Args:
            new_global_params: New global model parameters
        """
        # Compute deltas between new and previous parameters
        deltas = {}
        for name, param in new_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            deltas[name] = (param - prev_param) if prev_param is not None else param

        # Update velocity with momentum
        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    + (1.0 - self.momentum) * deltas[name]
                )

    def _velocity_l2_norm(self) -> float:
        """Calculate the L2 norm of the velocity vector.
        
        Returns:
            L2 norm of the velocity vector
        """
        total = 0.0
        for v in self.velocity.values():
            total += float(torch.sum(v.float() * v.float()).item())
        return float(np.sqrt(total))

    def _expected_position(self) -> Dict[str, torch.Tensor]:
        """Calculate the expected consensus position based on previous params and velocity.
        
        Returns:
            Dictionary of expected parameter values
        """
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel
        return expected_position

    def _calc_distances(self, client_params_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """Calculate distances between client parameters and expected consensus position.
        
        Args:
            client_params_list: List of client parameter dictionaries
            
        Returns:
            Array of distances for each client
        """
        # Expected position = previous params + velocity
        expected_position = self._expected_position()

        # Precompute flattened previous and velocity for delta-based metrics
        prev_flat = _flatten_state_dict(self.previous_global_params)
        vel_flat = _flatten_state_dict(self.velocity) if self.velocity else torch.zeros_like(prev_flat)
        vel_norm = float(torch.norm(vel_flat) + 1e-12)

        distances = []
        
        if self.distance_metric in ('l2', 'cosine'):
            # Existing logic (unchanged)
            if self.distance_metric == 'l2':
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
            else:  # cosine
                for client_params in client_params_list:
                    client_flat = _flatten_state_dict(client_params)
                    diff = client_flat - prev_flat
                    diff_norm = float(torch.norm(diff) + 1e-12)
                    cos_sim = float(torch.dot(diff, vel_flat) / (diff_norm * vel_norm))
                    distances.append(1.0 - cos_sim)
                return np.array(distances)

        elif self.distance_metric == 'update_cosine':
            # Cosine between client delta (w_i - w_prev) and velocity
            for client_params in client_params_list:
                client_flat = _flatten_state_dict(client_params)
                delta = client_flat - prev_flat
                delta_norm = float(torch.norm(delta) + 1e-12)
                cos_sim = float(torch.dot(delta, vel_flat) / (delta_norm * vel_norm))
                distances.append(1.0 - cos_sim)
            return np.array(distances)

        elif self.distance_metric == 'update_l2':
            for client_params in client_params_list:
                client_flat = _flatten_state_dict(client_params)
                delta = client_flat - prev_flat
                diff = delta - vel_flat
                distances.append(float(torch.norm(diff)))
            return np.array(distances)
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def _blend_with_samples(self, diversity_weights: np.ndarray, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> np.ndarray:
        """Blend diversity weights with client sample counts.
        
        Args:
            diversity_weights: Original diversity weights from distance-based calculation
            client_updates: List of (params, sample_count) tuples
            
        Returns:
            Blended weights
        """
        if self.blend_samples == 'none':
            return diversity_weights
            
        # Extract sample counts
        sample_counts = np.array([w for _, w in client_updates], dtype=float)
        total = float(sample_counts.sum())
        if total <= 0:
            return diversity_weights
            
        # Calculate sample fractions
        frac = sample_counts / total
        
        if self.blend_samples == 'sqrt':
            # Square root blending reduces the influence of large clients
            blend = np.sqrt(frac)
        elif self.blend_samples == 'linear':
            # Linear blending directly uses sample fractions
            blend = frac
        else:
            return diversity_weights
            
        # Multiply diversity weights by blend factors and renormalize
        blended = diversity_weights * blend
        s = blended.sum()
        return blended / s if s > 0 else diversity_weights

    def get_telemetry(self) -> Optional[Dict[str, float]]:
        """Get telemetry metrics from the last aggregation round.
        
        Returns:
            Dictionary of telemetry metrics, or None if not available
        """
        return self._last_telemetry

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using FedDive algorithm.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
            current_round: Current training round (used for dynamic temperature)
            total_rounds: Total number of rounds (used for dynamic temperature)
            
        Returns:
            Aggregated model parameters using diversity-weighted averaging
        """
        if not client_updates:
            logger.warning("[FedDive] No client updates received. Returning empty aggregation.")
            return {}
        
        # Update temperature at the start of aggregation (dynamic mode only)
        self._update_temperature(current_round, total_rounds)

        # Extract client parameter dicts
        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDive] The first client update is empty. Returning empty dict.")
            return {}

        num_clients = len(params_list)
        logger.info(f"[FedDive] Aggregating updates from {num_clients} clients (temperature={self.temperature:.4f}).")

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

        # Warmup: skip diversity weights in first N rounds
        if current_round < self.warmup_rounds:
            # use uniform or sample-based weights
            sample_counts = np.array([w for _, w in client_updates], dtype=float)
            if self.blend_samples in ('sqrt', 'linear') and sample_counts.sum() > 0:
                frac = sample_counts / sample_counts.sum()
                warm_weights = np.sqrt(frac) if self.blend_samples == 'sqrt' else frac
                warm_weights = warm_weights / warm_weights.sum()
            else:
                warm_weights = np.ones(num_clients, dtype=float) / num_clients

            aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
            for idx, (client_params, _) in enumerate(client_updates):
                w = float(warm_weights[idx])
                for name, tensor in client_params.items():
                    if name not in aggregated_params:
                        aggregated_params[name] = torch.zeros_like(tensor)
                    aggregated_params[name] += w * tensor

            logger.info(f"[FedDive] Using warmup weights in round {current_round} (warmup={self.warmup_rounds})")
            self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }
            self._last_telemetry = {
                "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
                "temperature": float(self.temperature),
                "velocity_l2": self._velocity_l2_norm(),
                "dist_mean_raw": float('nan'),
                "dist_std_raw": float('nan'),
                "dist_max_raw": float('nan'),
                "weight_entropy": float(-np.sum(warm_weights * (np.log(warm_weights + 1e-12))))
            }
            return aggregated_params

        # Step 3: Calculate distances from the expected consensus
        distances = self._calc_distances(params_list)
        raw_dist_mean = float(distances.mean()) if len(distances) > 0 else 0.0
        raw_dist_std = float(distances.std()) if len(distances) > 1 else 0.0
        raw_dist_max = float(distances.max()) if len(distances) > 0 else 0.0

        # Step 4: (Optional) Distance normalization
        if self.normalize_distances and len(distances) > 1:
            mean_val = float(distances.mean())
            std_val = float(distances.std())
            if std_val < self.epsilon:
                std_val = 1.0
            distances = (distances - mean_val) / std_val
            logger.debug(f"[FedDive] Normalized distances: {distances}")

        # Step 5: Temperature-scaled softmax for weighting
        scaled = distances / float(max(self.temperature, self.epsilon))
        # Subtract max for numerical stability
        scaled = scaled - np.max(scaled)
        exp_vals = np.exp(scaled)
        diversity_weights = exp_vals / np.sum(exp_vals)
        
        # Step 6: (Optional) Blend with sample counts
        diversity_weights = self._blend_with_samples(diversity_weights, client_updates)
        logger.debug(f"[FedDive] Final weights: {diversity_weights}")

        # Step 7: Weighted aggregation of parameters
        aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
        for idx, client_params in enumerate(params_list):
            w = float(diversity_weights[idx])
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        # Update previous global parameters for next round
        self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }

        # Calculate weight entropy for telemetry
        weight_entropy = float(-np.sum(diversity_weights * (np.log(diversity_weights + 1e-12))))
        self._last_telemetry = {
            "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
            "temperature": float(self.temperature),
            "velocity_l2": self._velocity_l2_norm(),
            "dist_mean_raw": raw_dist_mean,
            "dist_std_raw": raw_dist_std,
            "dist_max_raw": raw_dist_max,
            "weight_entropy": weight_entropy
        }

        logger.info("[FedDive] Aggregation complete.")
        return aggregated_params


class FedDiveR(FedDive):
    """Robust Federated Diversity Exploration (FedDiveR) aggregation strategy.
    
    A more robust variant of FedDive that uses median-based consensus tracking and
    outlier rejection to handle potential byzantine clients. Can also use dynamic
    temperature scheduling and supports the same distance metrics and sample blending
    options as FedDive.
    
    Reference: FedDiveR algorithm (Rateria et al., 2025)
    """
    
    def __init__(self, **kwargs):
        """Initialize FedDiveR aggregation strategy.
        
        Takes the same arguments as FedDive.
        """
        super().__init__(**kwargs)
        logger.info(f"FedDiveR initialized with {self.distance_metric} distance, "
                   f"{self.blend_samples} sample blending, and {self.warmup_rounds} warmup rounds")

    def _calc_median_params(self, client_params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute the parameter-wise median across client updates.
        
        Args:
            client_params_list: List of client parameter dictionaries
            
        Returns:
            Dictionary of parameter-wise median values
        """
        param_names = list(client_params_list[0].keys())
        median_params = {}
        
        for name in param_names:
            # Gather all client values for this parameter
            stack = []
            for cp in client_params_list:
                if name in cp:
                    stack.append(cp[name].unsqueeze(0))
            if not stack:
                logger.warning(f"[FedDiveR] Parameter {name} missing from all clients.")
                continue
                
            stacked_vals = torch.cat(stack, dim=0)
            # Compute median along client dimension (dim=0)
            median_tensor = torch.median(stacked_vals, dim=0).values
            median_params[name] = median_tensor
            
        return median_params

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using FedDiveR algorithm.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
            current_round: Current training round (used for dynamic temperature)
            total_rounds: Total number of rounds (used for dynamic temperature)
            
        Returns:
            Aggregated model parameters using robust diversity-weighted averaging
        """
        if not client_updates:
            logger.warning("[FedDiveR] No client updates received. Returning empty aggregation.")
            return {}
            
        # Update temperature if using dynamic mode
        self._update_temperature(current_round, total_rounds)

        # Extract client param dicts
        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDiveR] The first client update is empty. Returning empty dict.")
            return {}

        num_clients = len(params_list)
        logger.info(f"[FedDiveR] Aggregating updates from {num_clients} clients (temperature={self.temperature:.4f}).")

        # Warmup handling: identical to FedDive
        if current_round < self.warmup_rounds:
            sample_counts = np.array([w for _, w in client_updates], dtype=float)
            if self.blend_samples in ('sqrt', 'linear') and sample_counts.sum() > 0:
                frac = sample_counts / sample_counts.sum()
                warm_weights = np.sqrt(frac) if self.blend_samples == 'sqrt' else frac
                warm_weights = warm_weights / warm_weights.sum()
            else:
                warm_weights = np.ones(num_clients, dtype=float) / num_clients
                
            aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
            for idx, (client_params, _) in enumerate(client_updates):
                w = float(warm_weights[idx])
                for name, tensor in client_params.items():
                    if name not in aggregated_params:
                        aggregated_params[name] = torch.zeros_like(tensor)
                    aggregated_params[name] += w * tensor
                    
            logger.info(f"[FedDiveR] Using warmup weights in round {current_round} (warmup={self.warmup_rounds})")
            self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }
            self._last_telemetry = {
                "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
                "temperature": float(self.temperature),
                "velocity_l2": self._velocity_l2_norm(),
                "dist_mean_raw": float('nan'),
                "dist_std_raw": float('nan'),
                "dist_max_raw": float('nan'),
                "weight_entropy": float(-np.sum(warm_weights * (np.log(warm_weights + 1e-12)))),
                "outlier_count": 0.0,
                "fallback_median": 0.0
            }
            return aggregated_params

        # 1) Compute robust consensus (median) for momentum
        robust_global_params = self._calc_median_params(params_list)
        logger.debug("[FedDiveR] Computed robust consensus using parameter-wise median.")

        # If first round or mismatch in state, initialize
        if not self.velocity or not self.previous_global_params:
            self._init_state(robust_global_params)
            logger.debug("[FedDiveR] Initialized state vectors.")

        # 2) Update velocity with robust delta
        # Override _calc_velocity with direct implementation to use robust params
        deltas = {}
        for name, param in robust_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            deltas[name] = (param - prev_param) if prev_param is not None else param
        
        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    + (1.0 - self.momentum) * deltas[name]
                )
        
        logger.debug("[FedDiveR] Updated velocity using robust consensus.")

        # 3) Calculate distances from the expected consensus
        distances = self._calc_distances(params_list)
        raw_dist_mean = float(distances.mean()) if len(distances) > 0 else 0.0
        raw_dist_std = float(distances.std()) if len(distances) > 1 else 0.0
        raw_dist_max = float(distances.max()) if len(distances) > 0 else 0.0

        # 4) Detect outliers using robust z-score
        is_outlier = np.zeros(len(distances), dtype=bool)
        outlier_count = 0
        if len(distances) > 1:
            median_dist = np.median(distances)
            # MAD: Median Absolute Deviation
            mad = np.median(np.abs(distances - median_dist))
            # Robust Z-score with 0.6745 scaling factor (to make MAD comparable to std)
            robust_z_scores = np.abs(0.6745 * (distances - median_dist) / (mad + self.epsilon))
            is_outlier = robust_z_scores > 3.0
            outlier_count = int(np.sum(is_outlier))
            if outlier_count > 0:
                logger.info(f"[FedDiveR] Identified {outlier_count} outlier clients (indices: {np.where(is_outlier)[0]})")

        # 5) Normalize distances if enabled
        if self.normalize_distances and len(distances) > 1:
            mean_val = float(distances.mean())
            std_val = float(distances.std())
            if std_val < self.epsilon:
                std_val = 1.0
            distances = (distances - mean_val) / std_val
            logger.debug(f"[FedDiveR] Normalized distances: {distances}")

        # 6) Temperature-scaled softmax weighting
        scaled = distances / float(max(self.temperature, self.epsilon))
        # Subtract max for numerical stability
        scaled = scaled - np.max(scaled)
        exp_vals = np.exp(scaled)
        
        # Zero out outlier weights
        exp_vals[is_outlier] = 0.0

        sum_exp_vals = np.sum(exp_vals)
        if sum_exp_vals < self.epsilon:
            logger.warning("[FedDiveR] All clients were flagged as outliers or weights are zero. Falling back to median aggregation.")
            # Telemetry for fallback
            self._last_telemetry = {
                "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
                "temperature": float(self.temperature),
                "velocity_l2": self._velocity_l2_norm(),
                "dist_mean_raw": raw_dist_mean,
                "dist_std_raw": raw_dist_std,
                "dist_max_raw": raw_dist_max,
                "weight_entropy": 0.0,
                "outlier_count": float(outlier_count),
                "fallback_median": 1.0
            }
            return robust_global_params

        diversity_weights = exp_vals / sum_exp_vals
        
        # 7) Optional sample blending
        diversity_weights = self._blend_with_samples(diversity_weights, client_updates)
        logger.debug(f"[FedDiveR] Final weights (with outlier rejection): {diversity_weights}")

        # 8) Weighted aggregation
        aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
        for idx, client_params in enumerate(params_list):
            w = float(diversity_weights[idx])
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        # Update previous global parameters for next round
        self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }

        weight_entropy = float(-np.sum(diversity_weights * (np.log(diversity_weights + 1e-12))))
        self._last_telemetry = {
            "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
            "temperature": float(self.temperature),
            "velocity_l2": self._velocity_l2_norm(),
            "dist_mean_raw": raw_dist_mean,
            "dist_std_raw": raw_dist_std,
            "dist_max_raw": raw_dist_max,
            "weight_entropy": weight_entropy,
            "outlier_count": float(outlier_count),
            "fallback_median": 0.0
        }

        logger.info("[FedDiveR] Aggregation complete.")
        return aggregated_params


class FedDiveLW(FedDive):
    """FedDive-LayerWise: A refined version of FedDive that calculates divergence
    only on a specified subset of model layers (e.g., the classifier head).
    
    This variant applies diversity-based weights to the specified divergence layers
    while using sample-based weights for the remaining backbone layers.
    
    Like the base FedDive, supports distance metrics 'l2' or 'cosine', and
    sample blending options.
    """
    
    def __init__(self, divergence_layers: List[str], backbone_blend: str = 'sqrt', **kwargs):
        """Initialize FedDive-LayerWise aggregation strategy.
        
        Args:
            divergence_layers: List of layer names to use for divergence calculation
            backbone_blend: How to blend sample counts for backbone layers ('none', 'sqrt', 'linear')
            **kwargs: Additional arguments for FedDive
            
        Raises:
            ValueError: If divergence_layers is empty
        """
        super().__init__(**kwargs)
        if not divergence_layers:
            raise ValueError("FedDiveLW requires a list of 'divergence_layers'.")
        self.divergence_layers = divergence_layers
        self.backbone_blend = backbone_blend
        logger.info(f"[FedDive-LW] Initialized. Divergence will be computed on layers: {self.divergence_layers}")
        logger.info(f"[FedDive-LW] Backbone layers will use '{backbone_blend}' sample blending")

    def _expected_position(self) -> Dict[str, torch.Tensor]:
        """Override to calculate expected position only for specified layers.
        
        Returns:
            Dictionary of expected parameter values (only for divergence layers)
        """
        expected_position = {}
        for name in self.divergence_layers:
            if name in self.previous_global_params:
                prev_param = self.previous_global_params[name]
                vel = self.velocity.get(name, torch.zeros_like(prev_param))
                expected_position[name] = prev_param + vel
        return expected_position

    def _calc_distances(self, client_params_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """Calculate distances between client parameters and expected consensus position.
        
        Args:
            client_params_list: List of client parameter dictionaries
            
        Returns:
            Array of distances for each client
        """
        # Expected position = previous params + velocity
        expected_position = self._expected_position()

        # Precompute flattened previous and velocity for delta-based metrics
        prev_flat = _flatten_state_dict(self.previous_global_params)
        vel_flat = _flatten_state_dict(self.velocity) if self.velocity else torch.zeros_like(prev_flat)
        vel_norm = float(torch.norm(vel_flat) + 1e-12)

        distances = []
        
        if self.distance_metric in ('l2', 'cosine'):
            # Existing logic (unchanged)
            if self.distance_metric == 'l2':
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
            else:  # cosine
                for client_params in client_params_list:
                    client_flat = _flatten_state_dict(client_params)
                    diff = client_flat - prev_flat
                    diff_norm = float(torch.norm(diff) + 1e-12)
                    cos_sim = float(torch.dot(diff, vel_flat) / (diff_norm * vel_norm))
                    distances.append(1.0 - cos_sim)
                return np.array(distances)

        elif self.distance_metric == 'update_cosine':
            # Cosine between client delta (w_i - w_prev) and velocity
            for client_params in client_params_list:
                client_flat = _flatten_state_dict(client_params)
                delta = client_flat - prev_flat
                delta_norm = float(torch.norm(delta) + 1e-12)
                cos_sim = float(torch.dot(delta, vel_flat) / (delta_norm * vel_norm))
                distances.append(1.0 - cos_sim)
            return np.array(distances)

        elif self.distance_metric == 'update_l2':
            for client_params in client_params_list:
                client_flat = _flatten_state_dict(client_params)
                delta = client_flat - prev_flat
                diff = delta - vel_flat
                distances.append(float(torch.norm(diff)))
            return np.array(distances)
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using FedDiveLW algorithm.
        
        Applies diversity-based weights only to the specified divergence layers,
        while using sample-based weights for backbone layers.
        
        Args:
            client_updates: List of tuples containing (model_parameters, sample_count)
            current_round: Current training round (used for dynamic temperature)
            total_rounds: Total number of rounds (used for dynamic temperature)
            
        Returns:
            Aggregated model parameters using layer-specific weighting strategies
        """
        # Base validation and initialization
        if not client_updates:
            logger.warning("[FedDive-LW] No client updates received. Returning empty aggregation.")
            return {}
        
        self._update_temperature(current_round, total_rounds)
        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDive-LW] The first client update is empty. Returning empty dict.")
            return {}
            
        num_clients = len(params_list)
        logger.info(f"[FedDive-LW] Aggregating updates from {num_clients} clients (temperature={self.temperature:.4f}).")

        # Baseline mean for velocity update and initialization
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

        # Initialize velocity and previous params if needed
        if not self.velocity or not self.previous_global_params:
            self._init_state(new_global_params)
            logger.debug("[FedDive-LW] Initialized state vectors.")
            
        # Update velocity for tracking
        self._calc_velocity(new_global_params)
        logger.debug("[FedDive-LW] Updated velocity.")

        # Warmup handling: use sample-based weights for all layers
        if current_round < self.warmup_rounds:
            sample_counts = np.array([w for _, w in client_updates], dtype=float)
            if (self.backbone_blend in ('sqrt', 'linear')) and sample_counts.sum() > 0:
                frac = sample_counts / sample_counts.sum()
                warm = np.sqrt(frac) if self.backbone_blend == 'sqrt' else frac
                warm = warm / warm.sum()
            else:
                warm = np.ones(num_clients, dtype=float) / num_clients
                
            aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
            for idx, (client_params, _) in enumerate(client_updates):
                w = float(warm[idx])
                for name, tensor in client_params.items():
                    if name not in aggregated_params:
                        aggregated_params[name] = torch.zeros_like(tensor)
                    aggregated_params[name] += w * tensor
                    
            logger.info(f"[FedDive-LW] Using warmup weights in round {current_round} (warmup={self.warmup_rounds})")
            self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }
            self._last_telemetry = {
                "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
                "temperature": float(self.temperature),
                "velocity_l2": self._velocity_l2_norm(),
                "dist_mean_raw": float('nan'),
                "dist_std_raw": float('nan'),
                "dist_max_raw": float('nan'),
                "weight_entropy": float(-np.sum(warm * (np.log(warm + 1e-12))))
            }
            return aggregated_params

        # Calculate diversity weights for divergence layers
        distances = self._calc_distances(params_list)
        raw_dist_mean = float(distances.mean()) if len(distances) > 0 else 0.0
        raw_dist_std = float(distances.std()) if len(distances) > 1 else 0.0
        raw_dist_max = float(distances.max()) if len(distances) > 0 else 0.0
        
        if self.normalize_distances and len(distances) > 1:
            mean_val = float(distances.mean())
            std_val = float(distances.std())
            if std_val < self.epsilon:
                std_val = 1.0
            distances = (distances - mean_val) / std_val
            logger.debug(f"[FedDive-LW] Normalized distances: {distances}")

        scaled = distances / float(max(self.temperature, self.epsilon))
        scaled = scaled - np.max(scaled)
        exp_vals = np.exp(scaled)
        diversity_weights = exp_vals / np.sum(exp_vals)
        logger.debug(f"[FedDive-LW] Divergence layer weights: {diversity_weights}")

        # Calculate sample-based weights for backbone layers
        sample_counts = np.array([w for _, w in client_updates], dtype=float)
        if sample_counts.sum() > 0:
            frac = sample_counts / sample_counts.sum()
            if self.backbone_blend == 'sqrt':
                backbone_weights = np.sqrt(frac); backbone_weights /= backbone_weights.sum()
            elif self.backbone_blend == 'linear':
                backbone_weights = frac
            else:  # 'none' => pure uniform FedAvg-like (but should use sample frac for fairness)
                backbone_weights = frac
        else:
            backbone_weights = np.ones(num_clients, dtype=float) / num_clients
        logger.debug(f"[FedDive-LW] Backbone layer weights: {backbone_weights}")

        # Layer-specific weighted aggregation
        aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
        for idx, (client_params, _) in enumerate(client_updates):
            for name, tensor in client_params.items():
                if name in self.divergence_layers:
                    w = float(diversity_weights[idx])
                else:
                    w = float(backbone_weights[idx])
                
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        # Update previous global parameters for next round
        self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }

        # Telemetry
        weight_entropy = float(-np.sum(diversity_weights * (np.log(diversity_weights + 1e-12))))
        self._last_telemetry = {
            "is_dynamic": float(1.0 if self.is_dynamic else 0.0),
            "temperature": float(self.temperature),
            "velocity_l2": self._velocity_l2_norm(),
            "dist_mean_raw": raw_dist_mean,
            "dist_std_raw": raw_dist_std,
            "dist_max_raw": raw_dist_max,
            "weight_entropy": weight_entropy
        }

        logger.info("[FedDive-LW] Aggregation complete.")
        return aggregated_params