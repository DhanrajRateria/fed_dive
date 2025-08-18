"""
Federated learning client implementation.

This module provides the client-side functionality for federated learning,
handling local model training and communication with the server.
"""

import logging
import copy
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Configure logger for this module
logger = logging.getLogger(__name__)
# Example basic config (configure this globally in your main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FederatedClient:
    """
    Client in a federated learning system.

    Responsible for:
    - Holding local data and model.
    - Training the model on local data.
    - Calculating model updates (by returning updated parameters).
    - Receiving global model updates.
    - Evaluating the model.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_class: torch.optim.Optimizer = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        device: Optional[torch.device] = None # Added device handling
    ):
        """
        Initialize a federated learning client.

        Args:
            client_id: Unique identifier for the client.
            model: PyTorch model instance to train.
            dataset: Local dataset (PyTorch Dataset) for training.
            batch_size: Batch size for local training.
            learning_rate: Learning rate for the optimizer.
            optimizer_class: PyTorch optimizer class (e.g., torch.optim.SGD).
            optimizer_kwargs: Additional keyword arguments for the optimizer.
            loss_fn: PyTorch loss function instance.
            device: The torch.device (e.g., 'cpu', 'cuda') to run training on.
                    If None, attempts to use CUDA if available, else CPU.
        """
        self.client_id = client_id

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"Client {client_id}: No device specified, selected {self.device}.")
        else:
            self.device = device
            logger.debug(f"Client {client_id}: Using specified device {self.device}.")

        # Move model to device *before* initializing optimizer
        self.model = model

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            logger.debug(f"Client {client_id}: Moved nn.Module model to {self.device}.")
            # Setup optimizer only for nn.Module
            opt_cls = optimizer_class or torch.optim.SGD # Default if needed
            loss_func = loss_fn or nn.CrossEntropyLoss() # Default if needed
            self.optimizer = opt_cls(
                self.model.parameters(),
                lr=learning_rate,
                **(optimizer_kwargs or {})
            )
            self.loss_fn = loss_func
        elif hasattr(self.model, 'params') and isinstance(self.model.params, np.ndarray):
             # Example: VQC - No PyTorch optimizer/loss needed here
             logger.debug(f"Client {client_id}: Model appears NumPy-based (like VQC). Skipping .to(device) and PyTorch optimizer setup.")
             self.optimizer = None
             self.loss_fn = None # VQC uses internal loss calculation
        else:
             logger.warning(f"Client {client_id}: Model type {type(self.model)} not nn.Module or VQC-like. Device placement/optimizer skipped.")
             self.optimizer = None
             self.loss_fn = None

        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Setup data loader
        # Consider adding num_workers and pin_memory for performance if using GPU
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        logger.info(f"Client {self.client_id} initialized on device '{self.device}' "
                    f"with {len(dataset)} samples.")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the current model parameters (weights) as a state dictionary.
        Parameters are returned on the CPU.

        Returns:
            OrderedDict containing model state_dict (parameter tensors on CPU).
        """
        logger.debug(f"Client {self.client_id}: Getting parameters.")
        # Return state_dict for easier loading, ensure tensors are on CPU
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set parameters from state_dict (for nn.Module). Base implementation."""
        if isinstance(self.model, nn.Module):
            logger.debug(f"Client {self.client_id}: Setting nn.Module parameters (state_dict).")
            try:
                self.model.load_state_dict(parameters)
                logger.debug(f"Client {self.client_id}: nn.Module parameters updated successfully.")
            except RuntimeError as e:
                 logger.error(f"Client {self.client_id}: Error setting nn.Module parameters: {e}", exc_info=True); raise
            except Exception as e:
                logger.error(f"Client {self.client_id}: Unexpected error setting nn.Module parameters: {e}", exc_info=True); raise
        else:
            logger.warning(f"Client {self.client_id}: set_parameters called on non-nn.Module. Operation skipped.")

    def train(
        self,
        epochs: int = 1,
        proximal_term: float = 0.0,
        global_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train the local model on the local dataset.

        Args:
            epochs: Number of local epochs to train for.
            proximal_term: Coefficient mu for the FedProx proximal term. If > 0,
                           global_params must be provided.
            global_params: State dictionary of the global model (parameters on CPU or client's device)
                           required for FedProx calculation.

        Returns:
            Tuple containing:
                - Updated model parameters (state dictionary on CPU).
                - Number of data samples processed during training (epochs * dataset size).
        """

        if not isinstance(self.model, nn.Module) or self.optimizer is None or self.loss_fn is None:
             logger.error(f"Client {self.client_id}: Cannot train model type {type(self.model)} using standard PyTorch train loop.")
             # Return current params and 0 samples, or raise error
             return self.get_parameters(), 0

        if proximal_term > 0 and global_params is None:
            raise ValueError("global_params must be provided when proximal_term > 0 for FedProx.")

        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs. "
                    f"Proximal term mu={proximal_term}.")
        self.model.train() # Set model to training mode

        # If using FedProx, keep a copy of global params on the correct device
        global_params_on_device = None
        if proximal_term > 0:
            global_params_on_device = {name: param.to(self.device) for name, param in global_params.items()}

        total_samples_processed = 0
        for epoch in range(epochs):
            running_loss = 0.0
            samples_in_epoch = 0
            num_batches = len(self.data_loader)

            for batch_idx, (data, target) in enumerate(self.data_loader):
                try:# Move data and target to the client's device
                    data, target = data.to(self.device), target.to(self.device)
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Error processing batch {batch_idx}: {e}. Ensure dataset yields (data, target).", exc_info=True)
                    continue
                current_batch_size = data.shape[0]
                samples_in_epoch += current_batch_size

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)

                # Add FedProx term if applicable
                if proximal_term > 0 and global_params_on_device is not None:
                    proximal_loss = 0.0
                    # Iterate through local model parameters
                    for name, local_param in self.model.named_parameters():
                        if local_param.requires_grad: # Only compute for trainable params
                            # Get corresponding global parameter (already on device)
                            global_param = global_params_on_device[name]
                            # Calculate squared norm difference
                            proximal_loss += torch.sum(torch.pow(local_param - global_param, 2))
                    loss += (proximal_term / 2.0) * proximal_loss

                # Backward pass and optimize
                loss.backward()
                # Optional: Gradient clipping can be added here if needed
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item() * current_batch_size # Accumulate loss weighted by batch size
                logger.debug(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, "
                             f"Batch Loss: {loss.item():.4f}")

            epoch_loss = running_loss / samples_in_epoch if samples_in_epoch > 0 else 0.0
            total_samples_processed += samples_in_epoch
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs} completed. Average Loss: {epoch_loss:.4f}")

        final_params = self.get_parameters() # Get params on CPU
        logger.info(f"Client {self.client_id}: Local training finished. Processed {total_samples_processed} samples.")
        # Return updated parameters and total samples processed
        return final_params, total_samples_processed

    def evaluate(self, val_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the current local model on a validation dataset.

        Args:
            val_dataset: PyTorch Dataset to use for evaluation. If None, uses the client's
                         training dataset for evaluation.

        Returns:
            Dictionary containing evaluation metrics (e.g., 'loss', 'accuracy').
        """
        eval_dataset = val_dataset if val_dataset is not None else self.dataset
        if len(eval_dataset) == 0:
            logger.warning(f"Client {self.client_id}: Evaluation dataset is empty. Returning zero metrics.")
            return {'loss': 0.0, 'accuracy': 0.0}

        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        logger.info(f"Client {self.client_id}: Starting evaluation on {len(eval_dataset)} samples.")

        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad(): # Disable gradient calculations
            for data, target in eval_loader:
                try:# Move data and target to the client's device
                    data, target = data.to(self.device), target.to(self.device)
                except Exception as e:
                    logger.error(f"Client {self.client_id}: Error processing eval batch: {e}", exc_info=True)
                    continue
                current_batch_size = data.shape[0]

                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item() * current_batch_size # Accumulate loss weighted by batch size

                # Calculate accuracy (assuming classification task)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total_samples += current_batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
            # Add other metrics here if needed (e.g., precision, recall)
        }

        logger.info(f"Client {self.client_id} [PyTorch]: Evaluation finished. Metrics: {metrics}")
        return metrics
    
# --- Utility: Noisy Client for Robustness Experiments ---
class NoisyFederatedClient(FederatedClient):
    """A client that adds significant Gaussian noise to its parameters after training."""
    def __init__(self, noise_level: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        logger.info(f"Client {self.client_id} is a NOISY client with noise level: {self.noise_level}")

    def train(self, *args, **kwargs) -> Tuple[Dict[str, torch.Tensor], int]:
        updated_params, num_samples = super().train(*args, **kwargs)
        if self.noise_level > 0 and num_samples > 0:
            logger.warning(f"Client {self.client_id} is adding noise to its parameters.")
            with torch.no_grad():
                for name in updated_params:
                    noise = torch.randn_like(updated_params[name]) * self.noise_level
                    updated_params[name] += noise
        return updated_params, num_samples