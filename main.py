"""
Main entry point for running FedDive experiments.
"""

import os
import logging
import argparse
import yaml
from typing import List

def setup_directories():
    """Create necessary directories for experiments."""
    dirs = [
        "results",
        "results/baseline_iid",
        "results/non_iid_performance",
        "results/temperature_study",
        "results/adversarial_test",
        "data"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_experiment(experiment_name: str, config_path: str = None):
    """Run a specific experiment."""
    if config_path is None:
        config_path = f"configs/{experiment_name}.yaml"
    
    if experiment_name == "baseline_iid":
        from experiments.baseline_iid import run_experiment
        run_experiment(config_path)
    elif experiment_name == "non_iid_performance":
        from experiments.non_iid_performance import run_experiment
        run_experiment(config_path)
    elif experiment_name == "temperature_study":
        from experiments.temperature_study import run_experiment
        run_experiment(config_path)
    elif experiment_name == "adversarial_test":
        from experiments.adversarial_test import run_experiment
        run_experiment(config_path)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

def run_all_experiments():
    """Run all experiments in sequence."""
    experiments = [
        # "baseline_iid",
        # "non_iid_performance",
        "temperature_study",
        "adversarial_test"
    ]
    
    for exp in experiments:
        logging.info(f"===== Running experiment: {exp} =====")
        run_experiment(exp)
        logging.info(f"===== Completed experiment: {exp} =====")

def main():
    parser = argparse.ArgumentParser(description="Run FedDive experiments")
    
    # Add arguments
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="all",
        choices=["all", "baseline_iid", "non_iid_performance", "temperature_study", "adversarial_test"],
        help="Experiment to run (default: all)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to config file (default: configs/<experiment_name>.yaml)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create directories
    setup_directories()
    
    # Run experiments
    if args.experiment == "all":
        run_all_experiments()
    else:
        run_experiment(args.experiment, args.config)

if __name__ == "__main__":
    main()