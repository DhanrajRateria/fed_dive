"""
Federated learning server implementation.
"""

import logging
import copy
import time
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Assuming these modules are correctly placed relative to this file
from .aggregation import AggregationStrategy, FedAvg
from .client import FederatedClient  # Assuming client is in the same directory

logger = logging.getLogger(__name__)
# Example basic config (configure this globally in your main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FederatedServer:
    """Federated Learning Server."""

    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        evaluation_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None,
        server_optimizer: Optional[str] = None,
        server_lr: float = 0.01
    ):
        """
        Initialize the FederatedServer.

        Args:
            model: Initial global PyTorch model instance.
            aggregation_strategy: Strategy for aggregating client updates (e.g., FedAvg).
                                   Defaults to FedAvg if None.
            evaluation_dataset: Optional PyTorch Dataset for server-side evaluation.
            device: The torch.device (e.g., 'cpu', 'cuda') for server-side operations
                    like evaluation. If None, attempts auto-detection.
        """
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"Server: No device specified, selected {self.device}.")
        else:
            self.device = device
            logger.debug(f"Server: Using specified device {self.device}.")

        # Move the global model to the server's designated device
        self.model = model

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            logger.debug(f"Server: Moved nn.Module global model to {self.device}.")
        elif hasattr(self.model, 'params') and isinstance(self.model.params, np.ndarray):
            # Example: VQC model - does not need moving, uses NumPy
            logger.debug("Server: Model appears NumPy-based (like VQC), not moving to device.")
        else:
            logger.warning(f"Server: Model type {type(self.model)} is not nn.Module. "
                           f"Device placement using .to() skipped.")

        self.aggregation_strategy = aggregation_strategy or FedAvg()
        self.clients: Dict[str, FederatedClient] = {}
        self.evaluation_dataset = evaluation_dataset
        self.round_history: List[Dict[str, Any]] = []

        self.server_optimizer = None
        if server_optimizer:
            if server_optimizer.lower() == 'adam':
                self.server_optimizer = torch.optim.Adam(self.model.parameters(), lr=server_lr)
                logger.info(f"Server-side Adam optimizer initialized with lr={server_lr}.")
            elif server_optimizer.lower() == 'sgd':
                self.server_optimizer = torch.optim.SGD(self.model.parameters(), lr=server_lr)
                logger.info(f"Server-side SGD optimizer initialized with lr={server_lr}.")

        logger.info(f"Federated server initialized on device '{self.device}' "
                    f"with aggregation strategy: {type(self.aggregation_strategy).__name__}")

    def register_client(self, client: FederatedClient) -> None:
        """Register a client instance with the server."""
        if not isinstance(client, FederatedClient):
            logger.error(f"Attempted to register invalid object as client: {type(client)}")
            raise TypeError("Registered object must be an instance of FederatedClient")
        if client.client_id in self.clients:
            logger.warning(f"Client {client.client_id} already registered. Overwriting existing entry.")
        self.clients[client.client_id] = client
        logger.info(f"Registered client '{client.client_id}'. Total clients: {len(self.clients)}.")

    def select_clients(self, fraction: float = 1.0, min_clients: int = 1) -> List[str]:
        """Select a random subset of registered clients."""
        import random

        available_client_ids = list(self.clients.keys())
        num_available = len(available_client_ids)

        if num_available == 0:
            logger.warning("Client selection requested, but no clients are registered.")
            return []

        # Calculate number of clients to select
        num_to_select = int(fraction * num_available)
        num_to_select = max(min_clients, num_to_select)  # Ensure minimum number
        num_to_select = min(num_to_select, num_available)  # Cannot select more than available

        selected_ids = random.sample(available_client_ids, num_to_select)
        logger.info(f"Selected {len(selected_ids)} clients out of {num_available} available "
                    f"(fraction={fraction}, min_clients={min_clients}): {selected_ids}")
        return selected_ids

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get the current global model parameters (state dictionary on CPU)."""
        logger.debug("Server: Getting global model parameters.")
        params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()} \
            if isinstance(self.model, nn.Module) else {}  # Handle non-pytorch models gracefully
        return params
    
    def _get_param_size(self, params_dict: Dict[str, torch.Tensor]) -> int:
        """Calculates total number of elements in a state_dict."""
        return sum(p.numel() for p in params_dict.values())

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Update global model with provided parameters (state dictionary)."""
        logger.debug("Server: Setting global model parameters.")
        if self.server_optimizer:
            self.server_optimizer.zero_grad()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in parameters:
                        pseudo_gradient = param.data - parameters[name].to(self.device)
                        param.grad = pseudo_gradient
            self.server_optimizer.step()
            logger.debug("Global model updated via server-side optimizer step.")
        else:
            try:
                # load_state_dict handles moving parameters to the model's device
                self.model.load_state_dict(parameters)
                logger.debug("Server: Global model parameters updated successfully.")
            except RuntimeError as e:
                logger.error(f"Server: Error setting global parameters. "
                            f"Mismatch in state_dict keys or shapes? Error: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Server: Unexpected error setting global parameters: {e}", exc_info=True)
                raise
            
    def train_round(
        self,
        current_round: int,  # Added current_round parameter
        total_rounds: int,   # Added total_rounds parameter
        client_ids: Optional[List[str]] = None,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        min_clients: int = 1,
        proximal_term: float = 0.0
    ) -> Dict[str, Any]:
        """
        Execute a single round of federated training.
        
        Args:
            current_round: The current training round (0-based index)
            total_rounds: The total number of training rounds
            client_ids: Optional specific clients to use, if None will select based on fraction
            local_epochs: Number of local training epochs on each client
            client_fraction: Fraction of total clients to select
            min_clients: Minimum number of clients to select
            proximal_term: FedProx regularization parameter (mu)
            
        Returns:
            Dict containing round metrics and information
        """
        round_start_time = time.time()
        logger.info(f"Starting federated training round {current_round + 1}/{total_rounds}...")

        # 1. Select clients
        if client_ids is None:
            selected_client_ids = self.select_clients(fraction=client_fraction, min_clients=min_clients)
        else:
            # Validate provided client IDs
            selected_client_ids = [cid for cid in client_ids if cid in self.clients]
            if len(selected_client_ids) != len(client_ids):
                logger.warning(f"Some provided client IDs not found: "
                             f"{set(client_ids) - set(selected_client_ids)}")
            logger.info(f"Using pre-selected clients: {selected_client_ids}")

        if not selected_client_ids:
            logger.warning("No clients selected or available for this training round.")
            return {"status": "failed", "error": "No clients selected", "duration_seconds": time.time() - round_start_time}

        # 2. Distribute global model parameters to selected clients
        logger.debug("Distributing global model to selected clients...")
        global_params_cpu = self.get_parameters()  # Get params on CPU

        download_bytes = np.sum([p.element_size() * p.numel() for p in global_params_cpu.values()])

        total_upload_bytes = 0

        client_updates = []
        participating_clients_info = {}  # Store client_id -> num_samples

        # 3. Trigger local training on selected clients
        for client_id in selected_client_ids:
            client = self.clients[client_id]
            logger.debug(f"--- Training Client: {client_id} ---")
            try:
                # Send global parameters (CPU tensors)
                client.set_parameters(global_params_cpu)

                # Trigger training (client handles moving params to its device)
                # Pass global_params_cpu needed for FedProx calculation client-side
                updated_params_cpu, num_samples = client.train(
                    epochs=local_epochs,
                    proximal_term=proximal_term,
                    global_params=global_params_cpu if proximal_term > 0 else None
                )
                # Estimate upload size (parameters received from client)
                total_upload_bytes += np.sum([p.element_size() * p.numel() for p in updated_params_cpu.values()])

                if num_samples > 0:
                    # Store update (parameters are already on CPU from client.train)
                    client_updates.append((updated_params_cpu, float(num_samples)))  # Use float for weight
                    participating_clients_info[client_id] = num_samples
                    logger.debug(f"Client {client_id}: Training complete. Samples={num_samples}.")
                else:
                    logger.warning(f"Client {client_id}: Trained on 0 samples. Excluding from aggregation.")

            except Exception as e:
                logger.error(f"Error during training for client {client_id}: {e}", exc_info=True)
                # Optionally remove client or mark as failed for this round

        # 4. Aggregate updates - now passing round information to aggregation strategy
        if not client_updates:
            logger.warning("No valid client updates received for aggregation.")
            return {"status": "failed", "error": "No client updates received", "duration_seconds": time.time() - round_start_time}

        logger.info(f"Aggregating updates from {len(client_updates)} clients "
                   f"using {type(self.aggregation_strategy).__name__}.")
        try:
            # Pass current_round and total_rounds to the aggregation strategy
            aggregated_params_cpu = self.aggregation_strategy.aggregate(
                client_updates, 
                current_round, 
                total_rounds
            )
        except Exception as e:
            logger.error(f"Error during aggregation: {e}", exc_info=True)
            return {"status": "failed", "error": "Aggregation failed", "duration_seconds": time.time() - round_start_time}

        # 5. Update global model
        # Aggregated params are on CPU, set_parameters handles moving to server device
        self.set_parameters(aggregated_params_cpu)
        logger.info("Global model updated with aggregated parameters.")

        # 6. Evaluate global model (optional)
        metrics = {}
        if self.evaluation_dataset is not None:
            logger.info("Evaluating updated global model...")
            try:
                metrics = self.evaluate()  # Evaluation uses self.device
            except Exception as e:
                logger.error(f"Error during global model evaluation: {e}", exc_info=True)
                metrics = {"eval_error": str(e)}

        # 7. Record round history
        round_duration = time.time() - round_start_time
        round_info = {
            "round": current_round + 1,  # Use the provided current_round + 1 for 1-based indexing
            "status": "success",
            "participating_clients": list(participating_clients_info.keys()),
            "client_samples": list(participating_clients_info.values()),
            "total_samples": sum(participating_clients_info.values()),
            "duration_seconds": round_duration,
            "evaluation_metrics": metrics,
            "comm_download_bytes_per_client": download_bytes if participating_clients_info else 0,
            "comm_total_upload_bytes": total_upload_bytes,
            "comm_avg_upload_bytes_per_client": total_upload_bytes / len(participating_clients_info) if participating_clients_info else 0
        }
        self.round_history.append(round_info)
        logger.info(f"Round {current_round + 1}/{total_rounds} completed in {round_duration:.2f}s. "
                   f"Eval Metrics: {metrics}")

        return round_info

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the global model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on. If None, uses self.evaluation_dataset
            
        Returns:
            Dict containing evaluation metrics (loss, accuracy)
        """
        eval_dataset = dataset if dataset is not None else self.evaluation_dataset
        if eval_dataset is None or len(eval_dataset) == 0:
            logger.warning("Server evaluation skipped: No data.")
            return {}

        # --- Evaluation only makes sense for PyTorch models here ---
        # --- QuantumFederatedServer overrides this for VQC ---
        if not isinstance(self.model, nn.Module):
            logger.warning(f"Server evaluation skipped: Global model type {type(self.model)} is not an nn.Module.")
            return {'loss': None, 'accuracy': None}  # Or empty dict

        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        logger.debug(f"Starting server-side evaluation (PyTorch Model) on {len(eval_dataset)} samples using device {self.device}.")
        self.model.eval()
        self.model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                current_batch_size = data.shape[0]
                output = self.model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * current_batch_size
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total_samples += current_batch_size
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        logger.debug(f"Server-side evaluation (PyTorch Model) completed. Metrics: {metrics}")
        return metrics

    def train(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        min_clients: int = 1,
        proximal_term: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple rounds of federated training.
        
        Args:
            num_rounds: Number of federation rounds to execute
            local_epochs: Number of local training epochs on each client
            client_fraction: Fraction of total clients to select per round
            min_clients: Minimum number of clients to select per round
            proximal_term: FedProx regularization parameter (mu)
            
        Returns:
            List of round history dictionaries containing metrics
        """
        logger.info(f"Starting federated training process for {num_rounds} rounds.")
        logger.info(f"Config: local_epochs={local_epochs}, client_fraction={client_fraction}, "
                   f"min_clients={min_clients}, FedProx mu={proximal_term}")

        # Initial evaluation before training
        if self.evaluation_dataset:
            logger.info("Performing initial evaluation of global model...")
            initial_metrics = self.evaluate()
            logger.info(f"Initial Evaluation Metrics: {initial_metrics}")

        for round_idx in range(num_rounds):
            logger.info(f"===== Starting Round {round_idx + 1}/{num_rounds} =====")
            round_metrics = self.train_round(
                current_round=round_idx,  # 0-based index for current round
                total_rounds=num_rounds,  # Pass total rounds to train_round
                local_epochs=local_epochs,
                client_fraction=client_fraction,
                min_clients=min_clients,
                proximal_term=proximal_term
            )
            # Logging for the round is done within train_round

            if round_metrics.get("status") == "failed":
                logger.error(f"Round {round_idx + 1} failed and will be skipped. Error: {round_metrics.get('error')}")
                # Decide if training should stop or continue
                # break

        logger.info(f"Federated training process completed after {num_rounds} rounds.")
        return self.round_history

    def save_model(self, path: str) -> None:
        """Save the global model's state dictionary."""
        logger.info(f"Saving global model state_dict to {path}")
        try:
            if isinstance(self.model, nn.Module):
                torch.save(self.model.state_dict(), path)
                logger.info("Global nn.Module model saved successfully.")
            # Add VQC saving logic if needed (handled by QuantumFederatedServer maybe?)
            # elif isinstance(self.model, VariationalQuantumClassifier):
            #      self.model.save_params(path.replace('.pt', '.npy')) # Adjust extension
            else:
                logger.warning(f"Model type {type(self.model)} not nn.Module, saving not implemented in base server.")
        except Exception as e:
            logger.error(f"Failed to save global model to {path}: {e}", exc_info=True)
            raise

    def load_model(self, path: str) -> None:
        """Load the global model from a state dictionary file."""
        logger.info(f"Loading global model state_dict from {path}")
        try:
            if isinstance(self.model, nn.Module):
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)  # Ensure on correct device
                logger.info("Global nn.Module model loaded successfully.")
            # Add VQC loading logic here?
            # elif isinstance(self.model, VariationalQuantumClassifier):
            #     self.model.load_params(path.replace('.pt', '.npy')) # Adjust extension
            else:
                logger.warning(f"Model type {type(self.model)} not nn.Module, loading not implemented in base server.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load global model from {path}: {e}", exc_info=True)
            raise