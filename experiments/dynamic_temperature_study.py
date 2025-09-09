"""
Dynamic Temperature Study:
Compares fixed vs dynamic temperature schedules for FedDive on non-IID MNIST/CIFAR.
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

    # Partition
    if config['dataset']['partition'] == 'dirichlet':
        client_datasets = FederatedDataset.dirichlet_partition(
            dataset=train_dataset,
            num_clients=config['federated']['num_clients'],
            num_classes=config['model']['num_classes'],
            alpha=config['dataset']['alpha']
        )
    else:
        client_datasets = FederatedDataset.iid_partition(train_dataset, config['federated']['num_clients'])

    results = {}

    out_dir = f"results/{config['experiment']['name']}"
    os.makedirs(out_dir, exist_ok=True)

    for run in range(config['experiment']['runs']):
        for setting in config['dynamic_settings']:
            label = setting['label']
            logger.info(f"Run {run} | Setting: {label}")

            # Create model
            model = create_model(config)

            # Create aggregator (Fixed or Dynamic FedDive)
            if setting.get('mode', 'dynamic') == 'fixed':
                aggregator = FedDive(
                    momentum=config['feddive']['momentum'],
                    epsilon=config['feddive']['epsilon'],
                    temperature=setting['temperature'],
                    normalize_distances=config['feddive']['normalize_distances'],
                    is_dynamic=False
                )
            else:
                aggregator = FedDive(
                    momentum=config['feddive']['momentum'],
                    epsilon=config['feddive']['epsilon'],
                    normalize_distances=config['feddive']['normalize_distances'],
                    is_dynamic=True,
                    temp_schedule=setting.get('schedule', 'cosine'),
                    initial_temp=setting.get('initial_temp', 5.0),
                    final_temp=setting.get('final_temp', 0.5)
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

            # Save CSV per setting
            df = pd.DataFrame({
                'round': list(range(1, len(history)+1)),
                'accuracy': round_accuracies,
                'loss': round_losses,
                'aggregator': label,
                'run': run
            })
            csv_path = os.path.join(out_dir, f"{label}_run{run}_history.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {csv_path}")

            final_acc = round_accuracies[-1] if round_accuracies else 0.0
            final_loss = round_losses[-1] if round_losses else 0.0
            if label not in results:
                results[label] = []
            results[label].append({'final_accuracy': final_acc, 'final_loss': final_loss})

    # Summary
    summary = {}
    for label, runs in results.items():
        fa = [r['final_accuracy'] for r in runs]
        fl = [r['final_loss'] for r in runs]
        summary[label] = {
            'accuracy_mean': float(np.mean(fa)),
            'accuracy_std': float(np.std(fa)),
            'loss_mean': float(np.mean(fl)),
            'loss_std': float(np.std(fl))
        }

    clean_summary = convert_numpy_to_python(summary)
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(clean_summary, f, indent=4)
    logger.info(f"Saved summary to {summary_path}")
    return summary

if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/dynamic_temperature_study.yaml")