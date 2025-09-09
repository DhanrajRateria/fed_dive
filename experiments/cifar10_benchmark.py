"""
CIFAR-10 Non-IID (Dirichlet) Benchmark to showcase FedDive supremacy:
- FedAvg (baseline)
- FedProx (baseline)
- FedDive-LW (classifier-only divergence) with cosine distance, sqrt sample blending
- FedDive-R (robust gating) with cosine distance, sqrt sample blending
- Linear temperature schedule (3.0 -> 1.5) for dynamic stability
- Client optimizer: SGD with momentum and weight decay (standard CIFAR-10 recipe)
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
from src.federated.aggregation import AggregationStrategy
from src.federated.utils import FederatedDataset, set_seed
from experiments.baseline_iid import load_dataset, create_model
from experiments.utils.json_utils import convert_numpy_to_python

logger = logging.getLogger(__name__)

def run_experiment(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    results = {}

    out_root = f"results/{config['experiment']['name']}"
    os.makedirs(out_root, exist_ok=True)

    for alpha in config['dataset']['alpha_values']:
        logger.info(f"Benchmark CIFAR-10 Non-IID with alpha={alpha}")
        results[str(alpha)] = {}

        for run in range(config['experiment']['runs']):
            train_dataset, test_dataset = load_dataset(config)

            client_datasets = FederatedDataset.dirichlet_partition(
                dataset=train_dataset,
                num_clients=config['federated']['num_clients'],
                num_classes=config['model']['num_classes'],
                alpha=alpha
            )

            for agg_cfg in config['aggregators']:
                agg_name = agg_cfg['name']
                logger.info(f"Aggregator: {agg_name}")

                if agg_name not in results[str(alpha)]:
                    results[str(alpha)][agg_name] = []

                model = create_model(config)
                aggregator = AggregationStrategy.from_config(agg_cfg)

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
                        learning_rate=config['federated']['learning_rate'],
                        optimizer_kwargs=config['federated'].get('optimizer_kwargs', {})
                    )
                    server.register_client(client)

                proximal_term = agg_cfg.get('mu', 0.0) if agg_name == 'fedprox' else 0.0

                history = server.train(
                    num_rounds=config['federated']['num_rounds'],
                    local_epochs=config['federated']['local_epochs'],
                    client_fraction=config['federated']['clients_per_round'] / config['federated']['num_clients'],
                    proximal_term=proximal_term
                )

                round_accuracies = [r['evaluation_metrics'].get('accuracy', 0.0) for r in history]
                round_losses = [r['evaluation_metrics'].get('loss', 0.0) for r in history]

                df = pd.DataFrame({
                    'round': list(range(1, len(history)+1)),
                    'accuracy': round_accuracies,
                    'loss': round_losses,
                    'aggregator': agg_name,
                    'alpha': alpha,
                    'run': run
                })
                out_dir = os.path.join(out_root, f"alpha_{alpha}")
                os.makedirs(out_dir, exist_ok=True)
                df.to_csv(os.path.join(out_dir, f"{agg_name}_run{run}_history.csv"), index=False)

                results[str(alpha)][agg_name].append({
                    'final_accuracy': round_accuracies[-1] if round_accuracies else 0.0,
                    'final_loss': round_losses[-1] if round_losses else 0.0
                })

    summary = {}
    for a, aggs in results.items():
        summary[a] = {}
        for agg_name, runs in aggs.items():
            fa = [r['final_accuracy'] for r in runs]
            fl = [r['final_loss'] for r in runs]
            summary[a][agg_name] = {
                'accuracy_mean': float(np.mean(fa)),
                'accuracy_std': float(np.std(fa)),
                'loss_mean': float(np.mean(fl)),
                'loss_std': float(np.std(fl)),
            }

    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy_to_python(summary), f, indent=4)
    logger.info(f"Saved summary to {summary_path}")
    return summary

if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/cifar10_benchmark.yaml")