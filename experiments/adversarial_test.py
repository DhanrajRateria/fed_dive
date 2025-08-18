"""
Experiment 4: Adversarial Resilience

This experiment tests the robustness of FedDive-R against noisy/adversarial clients
and compares it with standard FedDive and other robust aggregation methods.
"""

import os
import json
from experiments.utils.json_utils import convert_numpy_to_python
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from torchvision import datasets, transforms

from src.federated.client import FederatedClient, NoisyFederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import AggregationStrategy
from src.federated.utils import FederatedDataset, set_seed
from src.models.simple_cnn import SimpleCNN
from src.models.mlp import SimpleMLP
from experiments.utils.plot_utils import plot_adversarial_comparison

logger = logging.getLogger(__name__)

def load_dataset(config: Dict):
    """Load dataset based on config."""
    # Reuse load_dataset from baseline_iid.py
    from experiments.baseline_iid import load_dataset
    return load_dataset(config)

def create_model(config: Dict) -> torch.nn.Module:
    """Create model based on config."""
    # Reuse create_model from baseline_iid.py
    from experiments.baseline_iid import create_model
    return create_model(config)

def create_aggregator(config: Dict) -> AggregationStrategy:
    """Create aggregator based on config."""
    return AggregationStrategy.from_config(config)

def run_experiment(config_path: str) -> Dict:
    """Run the adversarial resilience experiment with the given configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Results structure: {aggregator: {run: metrics}}
    results = {}
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    
    # Create non-IID partition (moderate heterogeneity)
    client_datasets = FederatedDataset.dirichlet_partition(
        dataset=train_dataset,
        num_clients=config['federated']['num_clients'],
        num_classes=config['model']['num_classes'],
        alpha=config['dataset']['alpha']
    )
    
    # Run multiple independent trials
    for run in range(config['experiment']['runs']):
        logger.info(f"Starting run {run+1}/{config['experiment']['runs']}")
        
        # Test each aggregation strategy
        for agg_config in config['aggregators']:
            agg_name = agg_config['name']
            logger.info(f"Testing aggregator: {agg_name}")
            
            if agg_name not in results:
                results[agg_name] = []
            
            # Create fresh model
            model = create_model(config)
            
            # Create aggregator
            aggregator = create_aggregator(agg_config)
            
            # Setup server
            server = FederatedServer(
                model=model,
                aggregation_strategy=aggregator,
                evaluation_dataset=test_dataset
            )
            
            # Create and register clients (some noisy, some normal)
            num_noisy = config['federated']['noise_clients']
            noise_level = config['federated']['noise_level']
            
            for i in range(config['federated']['num_clients']):
                # First n clients are noisy
                if i < num_noisy:
                    logger.info(f"Creating noisy client {i} with noise level {noise_level}")
                    client = NoisyFederatedClient(
                        client_id=f"client_{i}_noisy",
                        model=create_model(config),
                        dataset=client_datasets[i],
                        batch_size=config['dataset']['batch_size'],
                        learning_rate=config['federated']['learning_rate'],
                        noise_level=noise_level
                    )
                else:
                    client = FederatedClient(
                        client_id=f"client_{i}",
                        model=create_model(config),
                        dataset=client_datasets[i],
                        batch_size=config['dataset']['batch_size'],
                        learning_rate=config['federated']['learning_rate']
                    )
                server.register_client(client)
            
            # Train
            history = server.train(
                num_rounds=config['federated']['num_rounds'],
                local_epochs=config['federated']['local_epochs'],
                client_fraction=config['federated']['clients_per_round'] / config['federated']['num_clients']
            )
            
            # Extract metrics
            round_accuracies = [round_info['evaluation_metrics'].get('accuracy', 0) 
                              for round_info in history]
            round_losses = [round_info['evaluation_metrics'].get('loss', 0) 
                          for round_info in history]
            
            # Store run results
            run_result = {
                'accuracies': round_accuracies,
                'losses': round_losses,
                'final_accuracy': round_accuracies[-1] if round_accuracies else 0,
                'final_loss': round_losses[-1] if round_losses else 0
            }
            
            results[agg_name].append(run_result)
            
            # Save detailed run data
            os.makedirs(f"results/{config['experiment']['name']}", exist_ok=True)
            # np.save(
            #     f"results/{config['experiment']['name']}/{agg_name}_run{run}.npy",
            #     run_result
            # )
            history_df = pd.DataFrame([
                {
                    'round': i + 1,
                    'accuracy': r['evaluation_metrics'].get('accuracy', np.nan),
                    'loss': r['evaluation_metrics'].get('loss', np.nan)
                } for i, r in enumerate(history)
            ])
            # Define the output path for the detailed CSV
            os.makedirs(f"results/{config['experiment']['name']}", exist_ok=True)
            csv_path = f"results/{config['experiment']['name']}/{agg_name}_run{run}_history.csv"
            history_df.to_csv(csv_path, index=False)
            logger.info(f"Saved detailed run history to {csv_path}")
    
    # Generate summary statistics
    summary = {}
    for agg_name in results:
        final_accs = [run['final_accuracy'] for run in results[agg_name]]
        final_losses = [run['final_loss'] for run in results[agg_name]]
        
        summary[agg_name] = {
            'accuracy_mean': np.mean(final_accs),
            'accuracy_std': np.std(final_accs),
            'loss_mean': np.mean(final_losses),
            'loss_std': np.std(final_losses)
        }
        
        logger.info(f"{agg_name}: "
                   f"Accuracy={summary[agg_name]['accuracy_mean']:.4f} ± "
                   f"{summary[agg_name]['accuracy_std']:.4f}")
    
    # Save summary
    clean_summary = convert_numpy_to_python(summary)
    
    # Save the clean summary as a human-readable JSON file
    summary_path = f"results/{config['experiment']['name']}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(clean_summary, f, indent=4)
    logger.info(f"Saved final summary to {summary_path}")

    # Create visualization
    plot_adversarial_comparison(
        summary,
        title=f"Robustness Against {num_noisy} Adversarial Clients (Noise Level: {noise_level})",
        filename=f"results/{config['experiment']['name']}/adversarial_comparison.png"
    )
    
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python adversarial_test.py <config_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    results = run_experiment(sys.argv[1])
    print("Experiment completed successfully.")