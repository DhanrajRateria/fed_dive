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
                blend_samples=str(config.get('blend_samples', 'none')).lower()
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
                blend_samples=str(config.get('blend_samples', 'none')).lower()
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
                momentum=float(config.get('momentum', 0.9)),
                epsilon=float(config.get('epsilon', 1e-8)),
                temperature=float(config.get('temperature', 1.0)),
                normalize_distances=bool(config.get('normalize_distances', True)),
                distance_metric=str(config.get('distance_metric', 'l2')).lower(),
                blend_samples=str(config.get('blend_samples', 'none')).lower(),
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

        parameters_list = [params for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates], dtype=np.float32)

        if np.any(weights < 0):
            raise ValueError("Client weights cannot be negative.")
        total_weight = weights.sum()
        normalized_weights = (np.ones(num_clients, dtype=np.float32) / num_clients) if total_weight <= 0 else weights / total_weight

        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("FedAvg: First client update dictionary is empty.")
            return {}
        aggregated_params = {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in first_client_params.items()}

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
        self.mu = float(mu)


class Median(AggregationStrategy):
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("Median: No client updates provided.")
            return {}
        parameters_list = [params for params, _ in client_updates]
        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("Median: First client update dictionary is empty.")
            return {}
        aggregated_params = {}
        for name in first_client_params.keys():
            tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
            if not tensors_to_stack:
                continue
            stacked_tensors = torch.stack(tensors_to_stack, dim=0)
            median_values = torch.median(stacked_tensors, dim=0).values
            aggregated_params[name] = median_values.to(first_client_params[name].dtype)
        return aggregated_params


class TrimmedMean(AggregationStrategy):
    def __init__(self, trim_ratio: float = 0.1):
        super().__init__()
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be between 0 (inclusive) and 0.5 (exclusive)")
        self.trim_ratio = float(trim_ratio)

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("TrimmedMean: No client updates provided.")
            return {}
        parameters_list = [params for params, _ in client_updates]
        k = int(np.floor(self.trim_ratio * num_clients))
        if 2 * k >= num_clients:
            return Median().aggregate(client_updates, current_round, total_rounds)
        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("TrimmedMean: First client update dictionary is empty.")
            return {}
        aggregated_params = {}
        for name in first_client_params.keys():
            tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
            if not tensors_to_stack:
                continue
            stacked_tensors = torch.stack(tensors_to_stack, dim=0)
            sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
            trimmed_tensors = sorted_tensors[k : num_clients - k]
            mean_values = torch.mean(trimmed_tensors, dim=0)
            aggregated_params[name] = mean_values.to(first_client_params[name].dtype)
        return aggregated_params


def _flatten_state_dict(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().flatten().float() for p in params.values()])


class FedDive(AggregationStrategy):
    """
    FedDive with optional:
    - distance_metric: 'l2' (default) or 'cosine' (directional divergence w.r.t velocity)
    - blend_samples: 'none' (default), 'sqrt', or 'linear' (post-softmax blend with sample fractions)
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
        blend_samples: str = 'none'
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be in [0,1).")
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.normalize_distances = bool(normalize_distances)

        self.is_dynamic = bool(is_dynamic)
        if self.is_dynamic:
            self.temp_schedule = temp_schedule
            self.initial_temp = float(initial_temp)
            self.final_temp = float(final_temp)
            self.temperature = self.initial_temp
        else:
            self.temperature = float(temperature)

        self.distance_metric = distance_metric  # 'l2' or 'cosine'
        self.blend_samples = blend_samples      # 'none' | 'sqrt' | 'linear'

        self.velocity: Dict[str, torch.Tensor] = {}
        self.previous_global_params: Dict[str, torch.Tensor] = {}
        self._last_telemetry: Optional[Dict[str, float]] = None

    def _update_temperature(self, current_round: int, total_rounds: int):
        if not self.is_dynamic or total_rounds <= 1:
            return
        progress = current_round / max(1, total_rounds - 1)
        if self.temp_schedule.lower() == 'cosine':
            cosine_val = 0.5 * (1 + np.cos(np.pi * progress))
            self.temperature = self.final_temp + (self.initial_temp - self.final_temp) * cosine_val
        elif self.temp_schedule.lower() == 'linear':
            self.temperature = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        else:
            raise ValueError(f"Unknown temperature schedule: {self.temp_schedule}")

    def _init_state(self, reference_params: Dict[str, torch.Tensor]):
        self.velocity = {name: torch.zeros_like(param) for name, param in reference_params.items()}
        self.previous_global_params = {name: param.clone() for name, param in reference_params.items()}

    def _calc_velocity(self, new_global_params: Dict[str, torch.Tensor]):
        deltas = {}
        for name, param in new_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            deltas[name] = (param - prev_param) if prev_param is not None else param
        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = self.momentum * self.velocity[name] + (1.0 - self.momentum) * deltas[name]

    def _velocity_l2_norm(self) -> float:
        total = 0.0
        for v in self.velocity.values():
            total += float(torch.sum(v.float() * v.float()).item())
        return float(np.sqrt(total))

    def _expected_position(self) -> Dict[str, torch.Tensor]:
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel
        return expected_position

    def _calc_distances(self, client_params_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        # L2 to expected position (default)
        expected_position = self._expected_position()
        distances = []
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

        elif self.distance_metric == 'cosine':
            # Compute angular divergence vs velocity: d_i = 1 - cos( (client - prev), velocity )
            # Flatten once for speed
            prev_flat = _flatten_state_dict(self.previous_global_params)
            vel_flat = _flatten_state_dict(self.velocity)
            vel_norm = float(torch.norm(vel_flat) + 1e-12)
            distances = []
            for client_params in client_params_list:
                client_flat = _flatten_state_dict(client_params)
                diff = client_flat - prev_flat
                diff_norm = float(torch.norm(diff) + 1e-12)
                cos_sim = float(torch.dot(diff, vel_flat) / (diff_norm * vel_norm))
                d = 1.0 - cos_sim  # in [0,2], larger => more directionally divergent
                distances.append(d)
            return np.array(distances)
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def _blend_with_samples(self, diversity_weights: np.ndarray, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> np.ndarray:
        if self.blend_samples == 'none':
            return diversity_weights
        sample_counts = np.array([w for _, w in client_updates], dtype=float)
        total = float(sample_counts.sum())
        if total <= 0:
            return diversity_weights
        frac = sample_counts / total
        if self.blend_samples == 'sqrt':
            blend = np.sqrt(frac)
        elif self.blend_samples == 'linear':
            blend = frac
        else:
            return diversity_weights
        blended = diversity_weights * blend
        s = blended.sum()
        return blended / s if s > 0 else diversity_weights

    def get_telemetry(self) -> Optional[Dict[str, float]]:
        return self._last_telemetry

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]], 
                 current_round: int = 0, total_rounds: int = 1) -> Dict[str, torch.Tensor]:
        if not client_updates:
            logger.warning("[FedDive] No client updates received. Returning empty aggregation.")
            return {}
        self._update_temperature(current_round, total_rounds)

        params_list = [cp for cp, _ in client_updates]
        if not params_list[0]:
            logger.warning("[FedDive] The first client update is empty. Returning empty dict.")
            return {}
        num_clients = len(params_list)

        # baseline mean for velocity update init
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

        if not self.velocity or not self.previous_global_params:
            self._init_state(new_global_params)

        self._calc_velocity(new_global_params)

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

        scaled = distances / float(max(self.temperature, self.epsilon))
        scaled = scaled - np.max(scaled)
        exp_vals = np.exp(scaled)
        diversity_weights = exp_vals / np.sum(exp_vals)

        # Optional sample blending
        diversity_weights = self._blend_with_samples(diversity_weights, client_updates)

        aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
        for idx, (client_params, _) in enumerate(client_updates):
            w = float(diversity_weights[idx])
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

        self.previous_global_params = { name: p.clone() for name, p in aggregated_params.items() }

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
        return aggregated_params


class FedDiveR(FedDive):
    """
    Robust variant with median-shielded velocity and MAD-based gating.
    Inherits distance_metric and blend_samples from FedDive.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calc_median_params(self, client_params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        param_names = list(client_params_list[0].keys())
        median_params = {}
        for name in param_names:
            stack = []
            for cp in client_params_list:
                if name in cp:
                    stack.append(cp[name].unsqueeze(0))
            if not stack:
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

        # Use robust delta for velocity
        deltas = {}
        for name, param in robust_global_params.items():
            prev_param = self.previous_global_params.get(name, None)
            deltas[name] = (param - prev_param) if prev_param is not None else param
        with torch.no_grad():
            for name in deltas:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(deltas[name])
                self.velocity[name] = self.momentum * self.velocity[name] + (1.0 - self.momentum) * deltas[name]

        distances = self._calc_distances(params_list)
        raw_dist_mean = float(distances.mean()) if len(distances) > 0 else 0.0
        raw_dist_std = float(distances.std()) if len(distances) > 1 else 0.0
        raw_dist_max = float(distances.max()) if len(distances) > 0 else 0.0

        # Robust gating via MAD
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
            # fallback to robust median consensus
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
        # Optional sample blending
        diversity_weights = self._blend_with_samples(diversity_weights, client_updates)

        aggregated_params = { name: torch.zeros_like(tensor) for name, tensor in params_list[0].items() }
        for idx, (client_params, _) in enumerate(client_updates):
            w = float(diversity_weights[idx])
            for name, tensor in client_params.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(tensor)
                aggregated_params[name] += w * tensor

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
        return aggregated_params


class FedDiveLW(FedDive):
    """
    Layer-wise divergence: compute divergence on a subset of layers (e.g., classifier head).
    Supports distance_metric 'l2' or 'cosine' and blend_samples.
    """
    def __init__(self, divergence_layers: List[str], **kwargs):
        super().__init__(**kwargs)
        if not divergence_layers:
            raise ValueError("FedDiveLW requires a list of 'divergence_layers'.")
        self.divergence_layers = divergence_layers
        logger.info(f"[FedDive-LW] Divergence on layers: {self.divergence_layers}")

    def _expected_position(self) -> Dict[str, torch.Tensor]:
        expected_position = {}
        for name, prev_param in self.previous_global_params.items():
            if name not in self.divergence_layers:
                continue
            vel = self.velocity.get(name, torch.zeros_like(prev_param))
            expected_position[name] = prev_param + vel
        return expected_position

    def _calc_distances(self, client_params_list: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        expected_position = self._expected_position()
        if self.distance_metric == 'l2':
            distances = []
            for client_params in client_params_list:
                dist_sq = 0.0
                for name in self.divergence_layers:
                    if name in expected_position and name in client_params:
                        diff = client_params[name] - expected_position[name]
                        dist_sq += torch.sum(diff * diff).item()
                distances.append(np.sqrt(dist_sq))
            return np.array(distances)
        elif self.distance_metric == 'cosine':
            # Flatten only selected layers
            prev_flat = torch.cat([self.previous_global_params[name].detach().flatten().float()
                                   for name in self.divergence_layers if name in self.previous_global_params])
            vel_flat = torch.cat([self.velocity.get(name, torch.zeros_like(self.previous_global_params[name])).detach().flatten().float()
                                  for name in self.divergence_layers if name in self.previous_global_params])
            vel_norm = float(torch.norm(vel_flat) + 1e-12)
            distances = []
            for client_params in client_params_list:
                client_flat = torch.cat([client_params[name].detach().flatten().float()
                                        for name in self.divergence_layers if name in client_params])
                diff = client_flat - prev_flat
                diff_norm = float(torch.norm(diff) + 1e-12)
                cos_sim = float(torch.dot(diff, vel_flat) / (diff_norm * vel_norm))
                distances.append(1.0 - cos_sim)
            return np.array(distances)
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")