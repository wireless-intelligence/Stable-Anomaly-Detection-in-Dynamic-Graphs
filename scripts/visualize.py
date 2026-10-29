import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.visualization import (plot_training_curves, plot_ablation_comparison,
                               plot_multi_dataset_comparison, plot_latency_memory,
                               plot_score_distribution, create_d3_visualization)
from src.utils import load_json

def parse_args():
    parser = argparse.ArgumentParser(description='SAG Visualization')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'ablation', 'comparison', 
                                'distribution', 'graph', 'all'])
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to results JSON file or directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/figures')
    parser.add_argument('--dataset', type=str, default='')
    
    parser.add_argument('--scores_path', type=str, default=None,
                        help='Path to anomaly scores .npy file (for distribution/graph)')
    parser.add_argument('--adj_path', type=str, default=None,
                        help='Path to adjacency matrix (for graph visualization)')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to labels .npy file')
    
    return parser.parse_args()

def visualize_training(results_path, output_dir, dataset_name):
    results = load_json(results_path)
    
    if 'runs' in results:
        runs = results['runs']
    elif 'run_results' in results:
        runs = results['run_results']
    else:
        print("No run history found in results file")
        return
    
    best_run_idx = 0
    best_auroc = 0
    for i, run in enumerate(runs):
        if 'best_metrics' in run and run['best_metrics']['auroc'] > best_auroc:
            best_auroc = run['best_metrics']['auroc']
            best_run_idx = i
    
    if 'history' in runs[best_run_idx]:
        history = runs[best_run_idx]['history']
        output_path = os.path.join(output_dir, f'{dataset_name}_training_curves.png')
        plot_training_curves(history, output_path, dataset_name)
        print(f"Training curves saved to {output_path}")

def visualize_ablation(results_path, output_dir, dataset_name):
    results = load_json(results_path)
    
    if 'ablation_results' in results:
        ablation_results = results['ablation_results']
    else:
        print("No ablation results found")
        return
    
    output_path = os.path.join(output_dir, f'{dataset_name}_ablation_comparison.png')
    plot_ablation_comparison(ablation_results, output_path, dataset_name)
    print(f"Ablation comparison saved to {output_path}")

def visualize_comparison(results_dir, output_dir):
    results_dict = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                data = load_json(filepath)
                if 'dataset' in data and 'aggregated_results' in data:
                    dataset_name = data['dataset']
                    results_dict[dataset_name] = data['aggregated_results']
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not results_dict:
        print("No valid results found for comparison")
        return
    
    output_path = os.path.join(output_dir, 'multi_dataset_comparison.png')
    plot_multi_dataset_comparison(results_dict, output_path)
    print(f"Multi-dataset comparison saved to {output_path}")
    
    output_path2 = os.path.join(output_dir, 'latency_memory.png')
    plot_latency_memory(results_dict, output_path2)
    print(f"Latency/Memory plot saved to {output_path2}")

def visualize_distribution(scores_path, labels_path, output_dir, dataset_name):
    scores = np.load(scores_path)
    labels = np.load(labels_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_score_distribution.png')
    plot_score_distribution(scores, labels, output_path, dataset_name)
    print(f"Score distribution saved to {output_path}")

def visualize_graph(adj_path, scores_path, labels_path, output_dir, dataset_name):
    import scipy.sparse as sp
    import scipy.io as sio
    
    if adj_path.endswith('.mat'):
        data = sio.loadmat(adj_path)
        if 'Network' in data:
            adj = sp.csr_matrix(data['Network'])
        elif 'A' in data:
            adj = sp.csr_matrix(data['A'])
        else:
            raise KeyError("Adjacency matrix not found in .mat file")
    elif adj_path.endswith('.npy'):
        adj = sp.csr_matrix(np.load(adj_path))
    elif adj_path.endswith('.npz'):
        adj = sp.load_npz(adj_path)
    else:
        raise ValueError(f"Unsupported adjacency format: {adj_path}")
    
    scores = np.load(scores_path)
    labels = np.load(labels_path)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_graph_visualization.html')
    create_d3_visualization(adj, scores, labels, output_path)
    print(f"Graph visualization saved to {output_path}")

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_name = args.dataset if args.dataset else 'SAG'
    
    if args.mode == 'training':
        visualize_training(args.results_path, args.output_dir, dataset_name)
    
    elif args.mode == 'ablation':
        visualize_ablation(args.results_path, args.output_dir, dataset_name)
    
    elif args.mode == 'comparison':
        visualize_comparison(args.results_path, args.output_dir)
    
    elif args.mode == 'distribution':
        if not args.scores_path or not args.labels_path:
            print("Error: --scores_path and --labels_path required for distribution mode")
            return
        visualize_distribution(args.scores_path, args.labels_path, args.output_dir, dataset_name)
    
    elif args.mode == 'graph':
        if not args.adj_path or not args.scores_path or not args.labels_path:
            print("Error: --adj_path, --scores_path, and --labels_path required for graph mode")
            return
        visualize_graph(args.adj_path, args.scores_path, args.labels_path, 
                       args.output_dir, dataset_name)
    
    elif args.mode == 'all':
        print("Running all visualizations...")
        
        try:
            visualize_training(args.results_path, args.output_dir, dataset_name)
        except Exception as e:
            print(f"Training visualization failed: {e}")
        
        try:
            visualize_ablation(args.results_path, args.output_dir, dataset_name)
        except Exception as e:
            print(f"Ablation visualization failed: {e}")
        
        if args.scores_path and args.labels_path:
            try:
                visualize_distribution(args.scores_path, args.labels_path, 
                                      args.output_dir, dataset_name)
            except Exception as e:
                print(f"Distribution visualization failed: {e}")
            
            if args.adj_path:
                try:
                    visualize_graph(args.adj_path, args.scores_path, args.labels_path,
                                   args.output_dir, dataset_name)
                except Exception as e:
                    print(f"Graph visualization failed: {e}")
    
    print("Visualization completed!")

if __name__ == '__main__':
    main()
