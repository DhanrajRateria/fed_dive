"""
Experiment 3: Temperature Parameter Study

This experiment analyzes the impact of the temperature parameter in FedDive
on model convergence and final performance.
"""

import os
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
import pandas as pd
import json
from experiments.utils.json_utils import convert_numpy_to_python
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import FedDive
from src.federated.utils import FederatedDataset, set_seed
from src.models.simple_cnn import SimpleCNN
from src.models.mlp import SimpleMLP
from experiments.utils.plot_utils import plot_temperature_heatmap

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

def run_experiment(config_path: str) -> Dict:
    """Run the temperature parameter study with the given configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Results structure: {temperature: {run: metrics}}
    results = {}
    
    # Load dataset once
    train_dataset, test_dataset = load_dataset(config)
    
    # Create non-IID partition using Dirichlet
    client_datasets = FederatedDataset.dirichlet_partition(
        dataset=train_dataset,
        num_clients=config['federated']['num_clients'],
        num_classes=config['model']['num_classes'],
        alpha=config['dataset']['alpha']
    )
    
    # Iterate through different temperature values
    for temp in config['feddive']['temperatures']:
        logger.info(f"Running experiments with temperature = {temp}")
        results[temp] = []
        
        # Run multiple independent trials
        for run in range(config['experiment']['runs']):
            logger.info(f"Starting run {run+1}/{config['experiment']['runs']} for temp={temp}")
            
            # Create fresh model
            model = create_model(config)
            
            # Create FedDive with specified temperature
            aggregator = FedDive(
                momentum=config['feddive']['momentum'],
                epsilon=config['feddive']['epsilon'],
                temperature=temp,
                normalize_distances=config['feddive']['normalize_distances']
            )
            
            # Setup server
            server = FederatedServer(
                model=model,
                aggregation_strategy=aggregator,
                evaluation_dataset=test_dataset
            )
            
            # Create and register clients
            for i in range(config['federated']['num_clients']):
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
            
            # Extract metrics progression
            round_accuracies = [round_info['evaluation_metrics'].get('accuracy', 0) 
                              for round_info in history]
            round_losses = [round_info['evaluation_metrics'].get('loss', 0) 
                          for round_info in history]
            
            # Store run results
            run_result = {
                'accuracies': round_accuracies,
                'losses': round_losses,
                'final_accuracy': round_accuracies[-1] if round_accuracies else 0,
                'final_loss': round_losses[-1] if round_losses else 0,
                'convergence_round': next((i for i, acc in enumerate(round_accuracies) 
                                        if acc >= 0.95 * max(round_accuracies)), 
                                       len(round_accuracies))
            }
            
            results[temp].append(run_result)
            
            # Save detailed run data
            os.makedirs(f"results/{config['experiment']['name']}", exist_ok=True)
            # np.save(
            #     f"results/{config['experiment']['name']}/temp_{temp}_run{run}.npy",
            #     run_result
            # )
            history_df = pd.DataFrame([
                {
                    'round': i + 1,
                    'accuracy': r['evaluation_metrics'].get('accuracy', np.nan),
                    'loss': r['evaluation_metrics'].get('loss', np.nan)
                } for i, r in enumerate(history)
            ])
            output_dir = f"results/{config['experiment']['name']}"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"temp_{temp}_run{run}_history.csv")
            history_df.to_csv(csv_path, index=False)
            logger.info(f"Saved detailed run history to {csv_path}")
    
    # Generate summary statistics
    summary = {}
    for temp in results:
        final_accs = [run['final_accuracy'] for run in results[temp]]
        convergence_rounds = [run['convergence_round'] for run in results[temp]]
        
        summary[temp] = {
            'accuracy_mean': np.mean(final_accs),
            'accuracy_std': np.std(final_accs),
            'convergence_round_mean': np.mean(convergence_rounds),
            'convergence_round_std': np.std(convergence_rounds)
        }
        
        logger.info(f"Temperature={temp}: "
                   f"Accuracy={summary[temp]['accuracy_mean']:.4f} ± "
                   f"{summary[temp]['accuracy_std']:.4f}, "
                   f"Convergence Round={summary[temp]['convergence_round_mean']:.1f}")
    
    # Save summary
    # with open(f"results/{config['experiment']['name']}/summary.yaml", 'w') as f:
    #     yaml.dump(summary, f)
    clean_summary = convert_numpy_to_python(summary)
    summary_path = f"results/{config['experiment']['name']}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(clean_summary, f, indent=4)
    logger.info(f"Saved final summary to {summary_path}")
    
    # Create visualization
    plot_temperature_heatmap(
        summary,
        title="Impact of Temperature Parameter on FedDive Performance",
        filename=f"results/{config['experiment']['name']}/temperature_heatmap.png"
    )
    
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python temperature_study.py <config_path>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    results = run_experiment(sys.argv[1])
    print("Experiment completed successfully.")