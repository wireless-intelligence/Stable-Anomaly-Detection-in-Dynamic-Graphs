import numpy as np
import torch
from .metrics import compute_all_metrics, aggregate_metrics, format_all_metrics

class Evaluator:
    def __init__(self):
        self.history = []
        self.best_metrics = None
        self.best_epoch = -1
    
    def reset(self):
        self.history = []
        self.best_metrics = None
        self.best_epoch = -1
    
    def evaluate(self, y_true, y_score, epoch=None):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.cpu().numpy()
        
        metrics = compute_all_metrics(y_true, y_score)
        metrics['epoch'] = epoch
        
        self.history.append(metrics)
        
        if self.best_metrics is None or metrics['auroc'] > self.best_metrics['auroc']:
            self.best_metrics = metrics.copy()
            self.best_epoch = epoch
        
        return metrics
    
    def get_best(self):
        return self.best_metrics, self.best_epoch
    
    def get_history(self):
        return self.history
    
    def get_metric_history(self, metric_name):
        return [m[metric_name] for m in self.history if metric_name in m]
    
    def summary(self):
        if not self.history:
            return {}
        
        return {
            'best_auroc': self.best_metrics['auroc'] if self.best_metrics else 0,
            'best_auprc': self.best_metrics['auprc'] if self.best_metrics else 0,
            'best_macro_f1': self.best_metrics['macro_f1'] if self.best_metrics else 0,
            'best_epoch': self.best_epoch,
            'num_epochs': len(self.history)
        }


class MultiRunEvaluator:
    def __init__(self, n_runs=10):
        self.n_runs = n_runs
        self.run_results = []
        self.current_run = 0
    
    def reset(self):
        self.run_results = []
        self.current_run = 0
    
    def add_run_result(self, metrics):
        self.run_results.append(metrics)
        self.current_run += 1
    
    def aggregate(self):
        if not self.run_results:
            return {}
        
        aggregated = aggregate_metrics(self.run_results)
        formatted = format_all_metrics(aggregated)
        
        return {
            'aggregated': aggregated,
            'formatted': formatted,
            'n_runs': len(self.run_results),
            'raw_results': self.run_results
        }
    
    def get_summary_string(self):
        agg = self.aggregate()
        if not agg:
            return "No results available"
        
        formatted = agg['formatted']
        return (f"AUROC: {formatted.get('auroc', 'N/A')} | "
                f"AUPRC: {formatted.get('auprc', 'N/A')} | "
                f"Macro-F1: {formatted.get('macro_f1', 'N/A')}")


class AblationEvaluator:
    def __init__(self, ablation_names):
        self.ablation_names = ablation_names
        self.evaluators = {name: MultiRunEvaluator() for name in ablation_names}
    
    def add_result(self, ablation_name, metrics):
        if ablation_name in self.evaluators:
            self.evaluators[ablation_name].add_run_result(metrics)
    
    def aggregate_all(self):
        results = {}
        for name, evaluator in self.evaluators.items():
            results[name] = evaluator.aggregate()
        return results
    
    def get_comparison_table(self):
        results = self.aggregate_all()
        table_data = []
        
        for name in self.ablation_names:
            if name in results and results[name]:
                agg = results[name].get('aggregated', {})
                row = {
                    'method': name,
                    'auroc_mean': agg.get('auroc', {}).get('mean', 0) * 100,
                    'auroc_std': agg.get('auroc', {}).get('std', 0) * 100,
                    'auprc_mean': agg.get('auprc', {}).get('mean', 0) * 100,
                    'auprc_std': agg.get('auprc', {}).get('std', 0) * 100,
                    'macro_f1_mean': agg.get('macro_f1', {}).get('mean', 0) * 100,
                    'macro_f1_std': agg.get('macro_f1', {}).get('std', 0) * 100,
                }
                table_data.append(row)
        
        return table_data
