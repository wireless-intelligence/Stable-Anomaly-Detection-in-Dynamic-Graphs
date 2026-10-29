import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'full': '#E74C3C',
    'no_stable': '#3498DB',
    'no_decay': '#2ECC71',
    'no_fairness': '#9B59B6',
    'BlogCatalog': '#3498DB',
    'Cora': '#E74C3C',
    'YelpReviews': '#2ECC71',
}

MARKERS = {
    'full': '*',
    'no_stable': 'o',
    'no_decay': 's',
    'no_fairness': 'D',
}

def plot_training_curves(history, save_path, dataset_name=''):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = [m['epoch'] for m in history['metrics']]
    auroc = [m['auroc'] for m in history['metrics']]
    auprc = [m['auprc'] for m in history['metrics']]
    macro_f1 = [m['macro_f1'] for m in history['metrics']]
    
    axes[0].plot(epochs, auroc, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title(f'{dataset_name} - AUROC', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, auprc, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUPRC', fontsize=12)
    axes[1].set_title(f'{dataset_name} - AUPRC', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, macro_f1, 'r-', linewidth=2, marker='^', markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Macro-F1', fontsize=12)
    axes[2].set_title(f'{dataset_name} - Macro-F1', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(ablation_results, save_path, dataset_name=''):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(ablation_results.keys())
    auroc_means = []
    auroc_stds = []
    macro_f1_means = []
    macro_f1_stds = []
    
    for method in methods:
        agg = ablation_results[method].get('aggregated', {})
        auroc_means.append(agg.get('auroc', {}).get('mean', 0) * 100)
        auroc_stds.append(agg.get('auroc', {}).get('std', 0) * 100)
        macro_f1_means.append(agg.get('macro_f1', {}).get('mean', 0) * 100)
        macro_f1_stds.append(agg.get('macro_f1', {}).get('std', 0) * 100)
    
    colors = [COLORS.get(m, '#333333') for m in methods]
    markers = [MARKERS.get(m, 'o') for m in methods]
    
    for i, method in enumerate(methods):
        axes[0].scatter(auroc_means[i], macro_f1_means[i], 
                       c=colors[i], marker=markers[i], s=200, 
                       label=method.replace('_', ' ').title(), zorder=5)
        axes[0].errorbar(auroc_means[i], macro_f1_means[i],
                        xerr=auroc_stds[i], yerr=macro_f1_stds[i],
                        fmt='none', c=colors[i], alpha=0.5, capsize=3)
    
    axes[0].set_xlabel('AUROC (%)', fontsize=12)
    axes[0].set_ylabel('Macro-F1 (%)', fontsize=12)
    axes[0].set_title(f'(a) Performance Comparison - {dataset_name}', fontsize=14)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = axes[1].bar(x - width, auroc_means, width, yerr=auroc_stds,
                        label='AUROC', color='#3498DB', capsize=3)
    bars2 = axes[1].bar(x, macro_f1_means, width, yerr=macro_f1_stds,
                        label='Macro-F1', color='#2ECC71', capsize=3)
    
    auprc_means = []
    auprc_stds = []
    for method in methods:
        agg = ablation_results[method].get('aggregated', {})
        auprc_means.append(agg.get('auprc', {}).get('mean', 0) * 100)
        auprc_stds.append(agg.get('auprc', {}).get('std', 0) * 100)
    
    bars3 = axes[1].bar(x + width, auprc_means, width, yerr=auprc_stds,
                        label='AUPRC', color='#F39C12', capsize=3)
    
    for bars, means in [(bars1, auroc_means), (bars2, macro_f1_means), (bars3, auprc_means)]:
        for bar, mean in zip(bars, means):
            axes[1].annotate(f'{mean:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)
    
    axes[1].set_ylabel('Performance (%)', fontsize=12)
    axes[1].set_title(f'(b) Multi-Metric Ablation - {dataset_name}', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multi_dataset_comparison(results_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for dataset_name, results in results_dict.items():
        auroc = results.get('auroc', {}).get('mean', 0) * 100
        macro_f1 = results.get('macro_f1', {}).get('mean', 0) * 100
        auroc_std = results.get('auroc', {}).get('std', 0) * 100
        macro_f1_std = results.get('macro_f1', {}).get('std', 0) * 100
        
        color = COLORS.get(dataset_name, '#333333')
        
        ax.scatter(auroc, macro_f1, c=color, s=200, label=dataset_name, zorder=5)
        ax.errorbar(auroc, macro_f1, xerr=auroc_std, yerr=macro_f1_std,
                   fmt='none', c=color, alpha=0.5, capsize=3)
    
    ax.set_xlabel('AUROC (%)', fontsize=14)
    ax.set_ylabel('Macro-F1 (%)', fontsize=14)
    ax.set_title('Cross-Dataset Performance Comparison', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_memory(results_dict, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = list(results_dict.keys())
    latencies = [results_dict[d].get('latency_ms', {}).get('mean', 0) for d in datasets]
    memories = [results_dict[d].get('memory_gb', {}).get('mean', 0) for d in datasets]
    
    colors = [COLORS.get(d, '#3498DB') for d in datasets]
    
    bars1 = axes[0].bar(datasets, latencies, color=colors)
    axes[0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0].set_title('Query Latency by Dataset', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, latencies):
        axes[0].annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    bars2 = axes[1].bar(datasets, memories, color=colors)
    axes[1].set_ylabel('Memory (GB)', fontsize=12)
    axes[1].set_title('Peak Memory Usage by Dataset', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, memories):
        axes[1].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores, labels, save_path, dataset_name=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    axes[0].hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='#3498DB', density=True)
    axes[0].hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='#E74C3C', density=True)
    axes[0].set_xlabel('Anomaly Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'{dataset_name} - Score Distribution', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
    axes[1].set_ylabel('Anomaly Score', fontsize=12)
    axes[1].set_title(f'{dataset_name} - Score Box Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
