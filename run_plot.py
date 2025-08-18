#!/usr/bin/env python3

import argparse
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def load_data(file_path):
    """Load data from YAML or JSON file."""
    suffix = Path(file_path).suffix.lower()
    with open(file_path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


def extract_metrics(data, metric_names=None):
    """Extract metrics from the data structure."""
    metrics = {}
    
    def extract_recursive(data, prefix=''):
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    extract_recursive(value, new_prefix)
                elif isinstance(value, (int, float)) and (metric_names is None or any(metric in new_prefix for metric in metric_names)):
                    metrics[new_prefix] = value
    
    extract_recursive(data)
    return metrics


def plot_comparison(data_files, metric_filter=None, output_dir='plots'):
    """Generate comparison plots from multiple data files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data files
    datasets = {}
    for file_path in data_files:
        name = Path(file_path).stem
        datasets[name] = load_data(file_path)
    
    # Extract metrics from all datasets
    all_metrics = {}
    for name, data in datasets.items():
        metrics = extract_metrics(data, metric_filter)
        all_metrics[name] = metrics
    
    # Group metrics by type
    grouped_metrics = {}
    for name, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            # Extract the base metric name (e.g., accuracy_mean, loss_mean)
            parts = metric_name.split('_')
            if len(parts) >= 2 and parts[-1] in ['mean', 'std']:
                base_name = parts[-2] + '_' + parts[-1]
            else:
                base_name = metric_name
                
            if base_name not in grouped_metrics:
                grouped_metrics[base_name] = {}
            grouped_metrics[base_name][name] = value
    
    # Create plots for each metric type
    for metric_name, values in grouped_metrics.items():
        plt.figure(figsize=(10, 6))
        
        # Sort items by dataset name for consistency
        sorted_items = sorted(values.items())
        datasets = [item[0] for item in sorted_items]
        metric_values = [item[1] for item in sorted_items]
        
        # Create bar chart
        plt.bar(datasets, metric_values)
        plt.title(f'{metric_name} Comparison')
        plt.ylabel(metric_name)
        plt.xlabel('Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(output_dir, f"{metric_name}_comparison.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()


def plot_algorithm_comparison(data_file, output_dir='plots'):
    """Generate comparison plots between different algorithms in a single file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_file)
    
    # Find all algorithms and metrics
    algorithms = set()
    metrics = set()
    
    def find_algorithms_and_metrics(data, prefix=''):
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                
                if 'accuracy_mean' in value:
                    # This is likely an algorithm entry
                    algorithms.add(key)
                    for metric_key in value.keys():
                        metrics.add(metric_key)
                else:
                    find_algorithms_and_metrics(value, new_prefix)
    
    find_algorithms_and_metrics(data)
    
    # Extract data for plotting
    plot_data = {}
    for metric in metrics:
        plot_data[metric] = {}
        
        def extract_metric_data(data, path=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    if key in algorithms and metric in value:
                        category = path if path else 'default'
                        if category not in plot_data[metric]:
                            plot_data[metric][category] = {}
                        plot_data[metric][category][key] = value[metric]
                    else:
                        extract_metric_data(value, new_path)
        
        extract_metric_data(data)
    
    # Create plots
    for metric, categories in plot_data.items():
        for category, alg_data in categories.items():
            plt.figure(figsize=(10, 6))
            
            # Sort items by algorithm name
            sorted_items = sorted(alg_data.items())
            algs = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create bar chart
            plt.bar(algs, values)
            plt.title(f'{metric} by Algorithm ({category})')
            plt.ylabel(metric)
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Clean up category name for filename
            clean_category = category.replace('.', '_').replace(' ', '_')
            output_file = os.path.join(output_dir, f"{clean_category}_{metric}_comparison.png")
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot data from converted YAML/JSON files')
    parser.add_argument('files', nargs='+', help='Input YAML or JSON files to plot')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory for plots')
    parser.add_argument('--metrics', '-m', nargs='+', help='Filter for specific metrics (e.g., accuracy, loss)')
    parser.add_argument('--algorithm-comparison', '-a', action='store_true', 
                       help='Create comparisons between algorithms within each file')
    
    args = parser.parse_args()
    
    if args.algorithm_comparison:
        for file_path in args.files:
            print(f"Generating algorithm comparisons for {file_path}...")
            plot_algorithm_comparison(file_path, args.output_dir)
    else:
        print(f"Generating comparison plots across {len(args.files)} files...")
        plot_comparison(args.files, args.metrics, args.output_dir)


if __name__ == "__main__":
    main()