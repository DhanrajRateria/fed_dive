"""
Stability Analysis:
Logs FedDive/FedDive-R telemetry (velocity norm, distance stats, weight entropy) and update norms across rounds.
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

def run_single_setting(config: Dict, agg_cfg: Dict, client_datasets, test_dataset, out_dir: str, run: int):
    agg_name = agg_cfg['name']
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
            learning_rate=config['federated']['learning_rate']
        )
        server.register_client(client)

    # Manual round loop to compute update norm
    prev_params = server.get_parameters()
    rows = []
    for r in range(config['federated']['num_rounds']):
        round_info = server.train_round(
            current_round=r,
            total_rounds=config['federated']['num_rounds'],
            local_epochs=config['federated']['local_epochs'],
            client_fraction=config['federated']['clients_per_round'] / config['federated']['num_clients'],
            min_clients=1,
            proximal_term=0.0
        )
        curr_params = server.get_parameters()

        # Compute update norm ||w_t - w_{t-1}||
        diff_sq = 0.0
        for k in prev_params.keys():
            a = prev_params[k].float()
            b = curr_params[k].float()
            diff_sq += float(torch.sum((b - a) * (b - a)).item())
        update_l2 = float(np.sqrt(diff_sq))
        prev_params = curr_params

        telemetry = round_info.get('aggregation_telemetry', {}) or {}
        rows.append({
            'round': round_info['round'],
            'accuracy': round_info['evaluation_metrics'].get('accuracy', np.nan),
            'loss': round_info['evaluation_metrics'].get('loss', np.nan),
            'update_l2': update_l2,
            'temperature': telemetry.get('temperature', np.nan),
            'velocity_l2': telemetry.get('velocity_l2', np.nan),
            'dist_mean_raw': telemetry.get('dist_mean_raw', np.nan),
            'dist_std_raw': telemetry.get('dist_std_raw', np.nan),
            'dist_max_raw': telemetry.get('dist_max_raw', np.nan),
            'weight_entropy': telemetry.get('weight_entropy', np.nan),
            'outlier_count': telemetry.get('outlier_count', np.nan),
            'fallback_median': telemetry.get('fallback_median', np.nan),
            'aggregator': agg_name
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{agg_name}_run{run}_stability.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved stability data to {csv_path}")

    final_row = rows[-1]
    return {
        'final_accuracy': float(final_row['accuracy']),
        'final_loss': float(final_row['loss']),
        'mean_update_l2': float(np.mean([r['update_l2'] for r in rows])),
        'std_update_l2': float(np.std([r['update_l2'] for r in rows])),
        'mean_velocity_l2': float(np.mean([r['velocity_l2'] for r in rows if not np.isnan(r['velocity_l2'])])),
    }

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

    summary = {}
    for agg_cfg in config['aggregators']:
        name = agg_cfg['name']
        summary[name] = []

        for run in range(config['experiment']['runs']):
            res = run_single_setting(config, agg_cfg, client_datasets, test_dataset, out_dir, run)
            summary[name].append(res)

    # Aggregate summary
    agg_summary = {}
    for name, runs in summary.items():
        accs = [r['final_accuracy'] for r in runs]
        losses = [r['final_loss'] for r in runs]
        update_means = [r['mean_update_l2'] for r in runs]
        velocity_means = [r['mean_velocity_l2'] for r in runs if not np.isnan(r['mean_velocity_l2'])]

        agg_summary[name] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'update_l2_mean': float(np.mean(update_means)),
            'update_l2_std': float(np.std(update_means)),
            'velocity_l2_mean': float(np.mean(velocity_means)) if velocity_means else np.nan
        }

    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(convert_numpy_to_python(agg_summary), f, indent=4)
    logger.info(f"Saved stability summary to results/{config['experiment']['name']}/summary.json")
    return agg_summary

if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/stability_analysis.yaml")