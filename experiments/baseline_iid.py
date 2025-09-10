"""
Experiment 1: Baseline Preservation (IID Data)

This experiment demonstrates that FedDive performs as well as FedAvg on IID data,
establishing that there's no performance penalty for using FedDive in standard scenarios.
"""

import os
import logging
import yaml
import torch
import pandas as pd
import json
from experiments.utils.json_utils import convert_numpy_to_python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import FedAvg, AggregationStrategy
from src.federated.utils import FederatedDataset, set_seed
from src.models.simple_cnn import SimpleCNN
from src.models.mlp import SimpleMLP
from experiments.utils.plot_utils import plot_training_curves

logger = logging.getLogger(__name__)

def load_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    """Load and prepare dataset based on config."""
    dataset_name = config['dataset']['name'].lower()
    
    if dataset_name == 'mnist':
        # MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar10':
        # CIFAR-10 dataset
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'synthetic':
        # Create synthetic dataset
        from sklearn.datasets import make_blobs
        from torch.utils.data import TensorDataset
        
        n_samples = 5000
        centers = 10  # Number of classes
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=config['experiment']['seed'])
        
        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        full_dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train and test
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return train_dataset, test_dataset

def create_model(config: Dict) -> torch.nn.Module:
    """Create model based on config."""
    model_name = config['model']['name'].lower()
    if model_name == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
    elif model_name == 'mlp':
        # For MNIST: 28*28 = 784
        # For CIFAR: 3*32*32 = 3072
        if config['dataset']['name'].lower() == 'mnist':
            input_dim = 784
        elif config['dataset']['name'].lower() == 'cifar10':
            input_dim = 3072
        elif config['dataset']['name'].lower() == 'synthetic':
            input_dim = 2  # 2D synthetic data
        else:
            raise ValueError(f"Unknown dataset for MLP: {config['dataset']['name']}")
            
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            num_classes=config['model']['num_classes']
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model

def create_aggregator(config: Dict) -> AggregationStrategy:
    """Create aggregator based on config."""
    from src.federated.aggregation import AggregationStrategy
    
    return AggregationStrategy.from_config(config)

def run_experiment(config_path: str) -> Dict:
    """Run the baseline IID experiment with the given configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Track results across runs
    all_results = {
        'fedavg': {'accuracy': [], 'loss': []},
        'feddive': {'accuracy': [], 'loss': []}
    }
    
    # Run multiple independent trials
    for run in range(config['experiment']['runs']):
        logger.info(f"Starting run {run+1}/{config['experiment']['runs']}")
        
        # Load dataset
        train_dataset, test_dataset = load_dataset(config)
        
        # Client partitioning (IID)
        client_datasets = FederatedDataset.iid_partition(
            train_dataset, 
            config['federated']['num_clients']
        )
        
        # Run experiment for each aggregation strategy
        for agg_config in config['aggregators']:
            agg_name = agg_config['name']
            logger.info(f"Running with {agg_name} aggregator")
            
            # Create fresh model for this run
            model = create_model(config)
            
            # Create aggregator
            aggregator = create_aggregator(agg_config)
            
            # Setup server with test dataset for evaluation
            server = FederatedServer(
                model=model,
                aggregation_strategy=aggregator,
                evaluation_dataset=test_dataset
            )
            
            # Create and register clients
            for i in range(config['federated']['num_clients']):
                client = FederatedClient(
                    client_id=f"client_{i}",
                    model=create_model(config),  # Each client gets a fresh model
                    dataset=client_datasets[i],
                    batch_size=config['dataset']['batch_size'],
                    learning_rate=config['federated']['learning_rate']
                )
                server.register_client(client)
            
            # Train for specified rounds
            history = server.train(
                num_rounds=config['federated']['num_rounds'],
                local_epochs=config['federated']['local_epochs'],
                client_fraction=config['federated']['clients_per_round'] / config['federated']['num_clients']
            )
            
            # Extract final accuracy
            final_accuracy = history[-1]['evaluation_metrics']['accuracy']
            final_loss = history[-1]['evaluation_metrics']['loss']
            
            # Store results
            all_results[agg_name]['accuracy'].append(final_accuracy)
            all_results[agg_name]['loss'].append(final_loss)
            
            # Get accuracy progression
            round_accuracies = [round_info['evaluation_metrics'].get('accuracy', 0) 
                              for round_info in history]
            round_losses = [round_info['evaluation_metrics'].get('loss', 0) 
                          for round_info in history]
            
            # Save results for this run
            os.makedirs(f"results/{config['experiment']['name']}", exist_ok=True)
            # np.save(
            #     f"results/{config['experiment']['name']}/{agg_name}_run{run}_metrics.npy",
            #     {
            #         'accuracy': round_accuracies,
            #         'loss': round_losses,
            #         'final_accuracy': final_accuracy,
            #         'final_loss': final_loss
            #     }
            # )
            history_df = pd.DataFrame([
                {
                    'round': i + 1,
                    'accuracy': r['evaluation_metrics'].get('accuracy', np.nan),
                    'loss': r['evaluation_metrics'].get('loss', np.nan)
                } for i, r in enumerate(history)
            ])
            # Define the output path for the detailed CSV
            output_dir = f"results/{config['experiment']['name']}"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"{agg_name}_run{run}_history.csv")
            history_df.to_csv(csv_path, index=False)
            logger.info(f"Saved detailed run history to {csv_path}")

    
    # Calculate and log average results
    results_summary = {}
    for agg_name in all_results:
        mean_acc = np.mean(all_results[agg_name]['accuracy'])
        std_acc = np.std(all_results[agg_name]['accuracy'])
        mean_loss = np.mean(all_results[agg_name]['loss'])
        std_loss = np.std(all_results[agg_name]['loss'])
        
        results_summary[agg_name] = {
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'loss_mean': mean_loss,
            'loss_std': std_loss
        }
        
        logger.info(f"{agg_name} - Accuracy: {mean_acc:.4f} ± {std_acc:.4f}, Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    
    # Save summary
    # with open(f"results/{config['experiment']['name']}/summary.yaml", 'w') as f:
    #     yaml.dump(results_summary, f)

    clean_summary = convert_numpy_to_python(results_summary)
    
    # Save the clean summary as a human-readable JSON file
    summary_path = f"results/{config['experiment']['name']}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(clean_summary, f, indent=4)
    logger.info(f"Saved final summary to {summary_path}")
    
    return results_summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python baseline_iid.py <config_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    results = run_experiment(sys.argv[1])
    print("Experiment completed successfully.")