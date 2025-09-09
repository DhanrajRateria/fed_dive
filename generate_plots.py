# generate_plots.py (Final, Separated Plots, PDF Output)
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(df_history: pd.DataFrame, title: str, filename: str, hue_col: str = 'aggregator'):
    """
    Plots accuracy and loss curves in SEPARATE figures from a combined history DataFrame.
    Generates two files: one for accuracy and one for loss.
    """
    if df_history.empty:
        print(f"  - SKIPPING PLOT '{title}': No detailed history data found.")
        return

    df_plot = df_history.copy()
    df_plot['label'] = df_plot[hue_col].apply(lambda x: str(x).upper().replace("FEDDIVER", "FEDDIVE-R"))
    
    base_name, _ = os.path.splitext(filename)
    accuracy_filename = f"{base_name}_accuracy.pdf"
    loss_filename = f"{base_name}_loss.pdf"

    # --- Plot 1: Accuracy Curve ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6)) # Adjusted size for a single plot
    
    sns.lineplot(data=df_plot, x='round', y='accuracy', hue='label', marker='.', errorbar='sd')
    
    plt.title(f'{title}\nTest Accuracy vs. Communication Round', fontsize=16, weight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.legend(title=hue_col.capitalize())
    plt.ylim(0, 1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(accuracy_filename, format='pdf', dpi=300)
    plt.close()
    print(f"  - Saved training curve plot: {accuracy_filename}")

    # --- Plot 2: Loss Curve ---
    plt.figure(figsize=(9, 6)) # Adjusted size for a single plot
    
    sns.lineplot(data=df_plot, x='round', y='loss', hue='label', marker='.', errorbar='sd')
    
    plt.title(f'{title}\nTest Loss vs. Communication Round', fontsize=16, weight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.legend(title=hue_col.capitalize())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(loss_filename, format='pdf', dpi=300)
    plt.close()
    print(f"  - Saved training curve plot: {loss_filename}")


def plot_summary_barplot(summary: dict, title: str, filename: str):
    """Creates a final accuracy bar plot from a summary dict."""
    if not summary: return
    data = [{'aggregator': algo.upper().replace("FEDDIVER", "FEDDIVE-R"), 'accuracy': m['accuracy_mean']} for algo, m in summary.items()]
    df = pd.DataFrame(data).sort_values('accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=df, x='aggregator', y='accuracy', hue='aggregator', palette='plasma', legend=False)
    plt.title(title, fontsize=16, weight='bold'); plt.xlabel('Aggregation Strategy'); plt.ylabel('Final Test Accuracy')
    plt.ylim(0, 1.05); plt.grid(axis='y', linestyle='--', alpha=0.7)
    for p in barplot.patches:
        barplot.annotate(f"{p.get_height():.4f}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0,9), textcoords='offset points')
    plt.tight_layout(); plt.savefig(filename, format='pdf', dpi=300); plt.close()
    print(f"  - Saved summary bar plot: {filename}")

def plot_non_iid_summary(summary: dict, filename: str, title: str = 'Non-IID Summary'):
    """Creates a grouped bar plot for the Non-IID experiment."""
    if not summary: return
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    data = [{'alpha': f"α={float(a)}",'aggregator': n.upper().replace("FEDDIVER", "FEDDIVE-R"),'accuracy': m['accuracy_mean']} for a,d in summary.items() for n,m in d.items()]
    df = pd.DataFrame(data)
    alpha_order = sorted([f"α={float(a)}" for a in summary.keys()], key=lambda x: float(x.split('=')[1]), reverse=True)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x='alpha', y='accuracy', hue='aggregator', palette='viridis', order=alpha_order)
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Data Distribution (Lower α is more Non-IID)'); plt.ylabel('Final Test Accuracy')
    min_accuracy = df['accuracy'].min()
    plt.ylim(bottom=max(0, min_accuracy - 0.02), top=1.0)
    plt.legend(title='Aggregator'); plt.tight_layout(); plt.savefig(filename, format='pdf', dpi=300); plt.close()
    print(f"  - Saved Non-IID summary plot: {filename}")

def plot_temperature_study(summary: dict, filename: str):
    if not summary: return
    import matplotlib.pyplot as plt
    temps = sorted([float(t) for t in summary.keys()])
    accuracies = [summary[str(t)]['accuracy_mean'] for t in temps]
    conv_rounds = [summary[str(t)]['convergence_round_mean'] for t in temps]
    fig, ax1 = plt.subplots(figsize=(10, 6)); color = 'tab:blue'
    ax1.set_xlabel('FedDive Temperature Parameter (Log Scale)'); ax1.set_ylabel('Final Test Accuracy', color=color)
    ax1.plot(temps, accuracies, color=color, marker='o', label='Accuracy'); ax1.tick_params(axis='y', labelcolor=color); ax1.set_xscale('log')
    ax2 = ax1.twinx(); color = 'tab:red'
    ax2.set_ylabel('Convergence Round (lower is faster)', color=color)
    ax2.plot(temps, conv_rounds, color=color, marker='x', linestyle='--', label='Convergence Round'); ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Temperature Mastery: Balancing Accuracy and Convergence', fontsize=16, weight='bold')
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best'); fig.tight_layout(); plt.savefig(filename, format='pdf', dpi=300); plt.close()
    print(f"  - Saved Temperature study plot: {filename}")

def plot_hyperparam_sensitivity(summary: dict, filename: str, param_name: str):
    if not summary: return
    import matplotlib.pyplot as plt
    param_values = sorted([float(p) for p in summary.keys()])
    accuracies = [summary[str(p)]['accuracy_mean'] for p in param_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, marker='o', linestyle='-')
    plt.title(f'FedDive Sensitivity to {param_name.capitalize()}', fontsize=16, weight='bold')
    plt.xlabel(f'{param_name.capitalize()} Value')
    plt.ylabel('Final Test Accuracy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"  - Saved {param_name} sensitivity plot: {filename}")

def main():
    results_dir = 'results'
    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found."); return

    for exp_name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_dir): continue
        print(f"\nProcessing Experiment: {exp_name.upper()}")
        all_history_dfs = []
        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file.endswith("_history.csv"):
                    try:
                        import pandas as pd
                        df = pd.read_csv(os.path.join(root, file))
                        parts = file.replace("_history.csv", "").split('_')
                        if parts[0] == 'temp':
                            df['aggregator'] = 'feddive'; df['temperature'] = float(parts[1])
                        elif parts[0] == 'momentum':
                            df['aggregator'] = 'feddive'; df['momentum'] = float(parts[1])
                        elif 'dynamic' in file:
                            df['aggregator'] = f"{parts[0]}_{parts[1]}_{parts[2]}"
                        else:
                            df['aggregator'] = parts[0]
                        
                        run_part = [p for p in parts if p.startswith('run')]
                        df['run'] = int(run_part[0].replace('run', '')) if run_part else 0
                        if 'alpha_' in root: df['alpha'] = float(root.split('alpha_')[-1])
                        all_history_dfs.append(df)
                    except Exception as e: print(f"  - WARNING: Could not process {file}: {e}")
        
        combined_history_df = pd.concat(all_history_dfs, ignore_index=True) if all_history_dfs else pd.DataFrame()

        summary = None
        summary_path = os.path.join(exp_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f: summary = json.load(f)
            print(f"  - Loaded summary from {summary_path}")

        if exp_name in ['adversarial_test', 'baseline_iid', 'dynamic_temperature_study']:
            title_map = {
                'adversarial_test': "Adversarial Resilience",
                'baseline_iid': "Baseline IID",
                'dynamic_temperature_study': "Dynamic Temperature on Non-IID"
            }
            title_prefix = title_map.get(exp_name, exp_name.replace('_', ' ').title())
            plot_training_curves(combined_history_df, f'{title_prefix}: Training Progression', os.path.join(exp_dir, f'{exp_name}_curves.pdf'))
            plot_summary_barplot(summary, f'{title_prefix}: Final Accuracy', os.path.join(exp_dir, f'{exp_name}_summary.pdf'))

        elif exp_name in ['non_iid_performance', 'cifar10_benchmark']:
            title = 'Non-IID on MNIST' if 'non_iid' in exp_name else 'Non-IID on CIFAR-10'
            plot_non_iid_summary(summary, os.path.join(exp_dir, f'{exp_name}_summary_barplot.pdf'), title=title)
            if not combined_history_df.empty and 'alpha' in combined_history_df.columns:
                df_alpha_01 = combined_history_df[combined_history_df['alpha'] == 0.1].copy()
                plot_training_curves(df_alpha_01, f'Performance Under Extreme Non-IID (α=0.1) on {title.split()[-1]}', os.path.join(exp_dir, f'{exp_name}_alpha_0.1_curves.pdf'))
            
        elif exp_name == 'temperature_study':
            plot_temperature_study(summary, os.path.join(exp_dir, 'temperature_study_tradeoff.pdf'))
            plot_training_curves(combined_history_df, 'Effect of Temperature on Training Progression', os.path.join(exp_dir, 'temperature_training_curves.pdf'), hue_col='temperature')
        
        elif exp_name == 'hyperparam_sensitivity':
            plot_hyperparam_sensitivity(summary, os.path.join(exp_dir, 'momentum_sensitivity.pdf'), param_name='momentum')
            if not combined_history_df.empty:
                plot_training_curves(combined_history_df, 'Effect of Momentum on Training Progression', os.path.join(exp_dir, 'momentum_training_curves.pdf'), hue_col='momentum')

    print("\nAnalysis complete. All plots saved in their respective experiment directories.")

if __name__ == '__main__':
    main()