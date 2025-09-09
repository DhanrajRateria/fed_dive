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
            # Check if dynamic temperature is enabled
            if config.get('is_dynamic', False):
                logger.info("Initializing FedDive with DYNAMIC temperature scheduling.")
                return FedDive(
                    momentum=float(config.get('momentum', 0.9)),
                    epsilon=float(config.get('epsilon', 1e-8)),
                    normalize_distances=bool(config.get('normalize_distances', True)),
                    is_dynamic=True,
                    temp_schedule=config.get('temp_schedule', 'cosine'),
                    initial_temp=float(config.get('initial_temp', 5.0)),
                    final_temp=float(config.get('final_temp', 0.5))
                )
            else:
                logger.info("Initializing FedDive with FIXED temperature.")
                return FedDive(
                    momentum=float(config.get('momentum', 0.9)),
                    epsilon=float(config.get('epsilon', 1e-8)),
                    temperature=float(config.get('temperature', 1.0)),
                    normalize_distances=bool(config.get('normalize_distances', True))
                )
        elif strategy_type == 'feddiver':
            # Check if dynamic temperature is enabled for FedDiveR
            if config.get('is_dynamic', False):
                logger.info("Initializing FedDiveR with DYNAMIC temperature scheduling.")
                return FedDiveR(
                    momentum=float(config.get('momentum', 0.9)),
                    epsilon=float(config.get('epsilon', 1e-8)),
                    normalize_distances=bool(config.get('normalize_distances', True)),
                    is_dynamic=True,
                    temp_schedule=config.get('temp_schedule', 'cosine'),
                    initial_temp=float(config.get('initial_temp', 5.0)),
                    final_temp=float(config.get('final_temp', 0.5))
                )
            else:
                logger.info("Initializing FedDiveR with FIXED temperature.")
                return FedDiveR(
                    momentum=float(config.get('momentum', 0.9)),
                    epsilon=float(config.get('epsilon', 1e-8)),
                    temperature=float(config.get('temperature', 1.0)),
                    normalize_distances=bool(config.get('normalize_distances', True))
                )
        elif strategy_type == 'feddive_lw':
            logger.info("Initializing FedDive-LayerWise (FedDive-LW).")
            return FedDiveLW(
                momentum=float(config.get('momentum', 0.9)),
                epsilon=float(config.get('epsilon', 1e-8)),
                temperature=float(config.get('temperature', 1.0)),
                normalize_distances=bool(config.get('normalize_distances', True)),
                divergence_layers=config.get('divergence_layers')
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
                logger.debug(f"Median: Aggregated parameter '{name}' shape: {aggregated_params[name].shape}")
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
                 logger.debug(f"TrimmedMean: Aggregated parameter '{name}' shape: {aggregated_params[name].shape} "
                              f"(trimmed {2*k}/{num_clients} values)")
            except Exception as e:
                 logger.error(f"TrimmedMean: Error aggregating parameter '{name}': {e}", exc_info=True)

        logger.info("TrimmedMean: Aggregation complete.")
        return aggregated_params


class FedDive(AggregationStrategy):
    """
    Federated Diversity Exploration (FedDive) aggregation strategy.
    
    FedDive uses momentum-based velocity tracking and distance-based weighting to enhance
    model exploration during federated learning. It can use either fixed or dynamic temperature
    scheduling to control exploration-exploitation trade-offs during training.
    
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
        final_temp: float = 0.5
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
            self.temperature = initial_temp  # Start with initial temperature
            logger.info(f"FedDive Dynamic Temp enabled: schedule={temp_schedule}, "
                        f"start={initial_temp}, end={final_temp}")
        else:
            self.temperature = float(temperature)
            logger.info(f"FedDive Fixed Temp: {self.temperature}")

        # Internal state variables that persist across rounds
        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}

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
            if prev_param is not None:
                deltas[name] = param - prev_param
            else:
                # If it doesn't exist, treat the entire param as delta
                deltas[name] = param

        # Update velocity with momentum
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
        """Calculate distances between client parameters and expected consensus position.
        
        Args:
            client_params_list: List of client parameter dictionaries
            
        Returns:
            Array of L2 distances for each client
        """
        # Expected consensus position is previous parameters plus velocity
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel

        # Calculate L2 distances for each client
        distances = []
        for client_params in client_params_list:
            dist_sq = 0.0
            for name, expected_val in expected_position.items():
                client_val = client_params.get(name, None)
                if client_val is not None:
                    diff = client_val - expected_val
                    dist_sq += torch.sum(diff * diff).item()
                else:
                    # If client doesn't have this parameter, add expected value's square
                    dist_sq += torch.sum(expected_val * expected_val).item()
            distances.append(np.sqrt(dist_sq))
            
        return np.array(distances)

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
        logger.info(f"[FedDive] Aggregating updates from {num_clients} clients "
                   f"(temperature={self.temperature:.4f}).")

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
    Robust Federated Diversity Exploration (FedDiveR) aggregation strategy.
    
    A more robust variant of FedDive that uses median-based consensus tracking and
    outlier rejection to handle potential byzantine clients. Can also use dynamic
    temperature scheduling.
    
    Reference: FedDiveR algorithm (Rateria et al., 2025)
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
        final_temp: float = 0.5
    ):
        """Initialize FedDiveR aggregation strategy.
        
        Args:
            momentum: Momentum coefficient for velocity tracking (0 <= momentum < 1)
            epsilon: Small constant for numerical stability
            temperature: Temperature parameter for softmax weighting (fixed mode only)
            normalize_distances: Whether to normalize distances before softmax
            is_dynamic: Whether to use dynamic temperature scheduling
            temp_schedule: Schedule type for dynamic temperature ('cosine' or 'linear')
            initial_temp: Starting temperature for dynamic scheduling
            final_temp: Final temperature for dynamic scheduling
            
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
            self.temperature = initial_temp  # Start with initial temperature
            logger.info(f"FedDiveR Dynamic Temp enabled: schedule={temp_schedule}, "
                        f"start={initial_temp}, end={final_temp}")
        else:
            self.temperature = float(temperature)
            logger.info(f"FedDiveR Fixed Temp: {self.temperature}")

        # Internal state variables
        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}

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
        
        logger.debug(f"[FedDiveR] Round {current_round}/{total_rounds}, "
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

    def _calc_median_params(
        self, client_params_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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

    def _calc_velocity(self, robust_global_params: Dict[str, torch.Tensor]):
        """Update velocity using robust consensus.
        
        Args:
            robust_global_params: Robust global parameters (from median)
        """
        # Compute deltas from robust global to previous global
        deltas = {}
        for name, param in robust_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            if prev_param is not None:
                deltas[name] = param - prev_param
            else:
                deltas[name] = param

        # Update velocity with momentum
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
        """Calculate distances between client parameters and expected consensus position.
        
        Args:
            client_params_list: List of client parameter dictionaries
            
        Returns:
            Array of L2 distances for each client
        """
        # Expected position = previous params + velocity
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel

        # Calculate L2 distances
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
        logger.info(f"[FedDiveR] Aggregating updates from {num_clients} clients "
                   f"(temperature={self.temperature:.4f}).")

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

        # 4) Detect outliers using robust z-score
        is_outlier = np.zeros(len(distances), dtype=bool)
        if len(distances) > 1:
            median_dist = np.median(distances)
            # MAD: Median Absolute Deviation
            mad = np.median(np.abs(distances - median_dist))
            # Robust Z-score with 0.6745 scaling factor (to make MAD comparable to std)
            robust_z_scores = np.abs(0.6745 * (distances - median_dist) / (mad + self.epsilon))
            
            # Identify outliers: anything beyond 3.0 z-score is considered an outlier
            is_outlier = robust_z_scores > 3.0 
            if np.any(is_outlier):
                logger.info(f"[FedDiveR] Identified {np.sum(is_outlier)} outlier clients "
                           f"(indices: {np.where(is_outlier)[0]})")

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
            logger.warning("[FedDiveR] All clients were flagged as outliers or weights are zero. "
                          "Falling back to median aggregation.")
            return robust_global_params

        diversity_weights = exp_vals / sum_exp_vals
        logger.debug(f"[FedDiveR] Diversity weights (with outlier rejection): {diversity_weights}")

        # 7) Weighted aggregation
        aggregated_params = {
            name: torch.zeros_like(tensor) for name, tensor in params_list[0].items()
        }
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

class FedDiveLW(FedDive):
    """
    FedDive-LayerWise: A refined version of FedDive that calculates divergence
    only on a specified subset of model layers (e.g., the classifier head).
    """
    def __init__(self, divergence_layers: List[str], **kwargs):
        super().__init__(**kwargs)
        if not divergence_layers:
            raise ValueError("FedDiveLW requires a list of 'divergence_layers'.")
        self.divergence_layers = divergence_layers
        logger.info(f"[FedDive-LW] Initialized. Divergence will be computed on layers: {self.divergence_layers}")

    def _calc_distances(self, client_params_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """
        Overridden method to calculate distances based only on specified layers.
        """
        # Expected consensus position
        expected_position = {
            name: self.previous_global_params[name] + self.velocity.get(name, 0)
            for name in self.divergence_layers
            if name in self.previous_global_params
        }

        distances = []
        for client_params in client_params_list:
            dist_sq = 0.0
            # Only iterate over the specified divergence layers
            for name in self.divergence_layers:
                if name in expected_position and name in client_params:
                    diff = client_params[name] - expected_position[name]
                    dist_sq += torch.sum(diff * diff).item()
            distances.append(np.sqrt(dist_sq))
        
        logger.debug(f"[FedDive-LW] Layer-wise distances computed: {distances}")
        return np.array(distances)