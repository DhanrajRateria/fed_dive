"""
Hyperparameter Sensitivity:
Study momentum (β) for FedDive on non-IID MNIST α=0.1.
"""

import os
import yaml
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import FedDive
from src.federated.utils import FederatedDataset, set_seed
from experiments.baseline_iid import load_dataset, create_model
from experiments.utils.json_utils import convert_numpy_to_python

logger = logging.getLogger(__name__)

def run_experiment(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    train_dataset, test_dataset = load_dataset(config)

    client_datasets = FederatedDataset.dirichlet_partition(
        dataset=train_dataset,
        num_clients=config['federated']['num_clients'],
        num_classes=config['model']['num_classes'],
        alpha=config['dataset']['alpha']
    )

    out_dir = f"results/{config['experiment']['name']}"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for momentum in config['feddive']['momentum_values']:
        results[str(momentum)] = []

        for run in range(config['experiment']['runs']):
            model = create_model(config)
            aggregator = FedDive(
                momentum=float(momentum),
                epsilon=config['feddive']['epsilon'],
                temperature=config['feddive']['temperature'],
                normalize_distances=config['feddive']['normalize_distances']
            )

            server = FederatedServer(
                model=model,
                aggregation_strategy=aggregator,
                evaluation_dataset=test_dataset
            )

            for i in range(config['federated']['num_clients']):
                client = FederatedClient(
                    client_id=f"client_{i}",
                    model=create_model(config),
                    dataset=client_datasets[i],
                    batch_size=config['dataset']['batch_size'],
                    learning_rate=config['federated']['learning_rate']
                )
                server.register_client(client)

            history = server.train(
                num_rounds=config['federated']['num_rounds'],
                local_epochs=config['federated']['local_epochs'],
                client_fraction=config['federated']['clients_per_round'] / config['federated']['num_clients']
            )

            round_accuracies = [r['evaluation_metrics'].get('accuracy', 0.0) for r in history]
            round_losses = [r['evaluation_metrics'].get('loss', 0.0) for r in history]

            df = pd.DataFrame({
                'round': list(range(1, len(history)+1)),
                'accuracy': round_accuracies,
                'loss': round_losses,
                'momentum': float(momentum),
                'run': run
            })
            df.to_csv(os.path.join(out_dir, f"momentum_{momentum}_run{run}_history.csv"), index=False)

            results[str(momentum)].append({
                'final_accuracy': round_accuracies[-1] if round_accuracies else 0.0,
                'final_loss': round_losses[-1] if round_losses else 0.0
            })

    summary = {}
    for m, runs in results.items():
        fa = [r['final_accuracy'] for r in runs]
        summary[m] = {
            'accuracy_mean': float(np.mean(fa)),
            'accuracy_std': float(np.std(fa))
        }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_to_python(summary), f, indent=4)
    logger.info(f"Saved summary to {summary_path}")
    return summary

if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/hyperparam_sensitivity.yaml")