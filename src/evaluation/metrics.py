import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, roc_curve)

def compute_auroc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.0

def compute_auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return 0.0

def compute_macro_f1(y_true, y_score, threshold=None):
    if threshold is None:
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0.0
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1
    else:
        y_pred = (y_score >= threshold).astype(int)
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

def compute_precision_at_k(y_true, y_score, k):
    indices = np.argsort(y_score)[::-1][:k]
    return np.sum(y_true[indices]) / k

def compute_recall_at_k(y_true, y_score, k):
    num_anomalies = np.sum(y_true)
    if num_anomalies == 0:
        return 0.0
    indices = np.argsort(y_score)[::-1][:k]
    return np.sum(y_true[indices]) / num_anomalies

def compute_balanced_auprc_f1(y_true, y_score, n_trials=20, seed=0):
    """CoLA/DOMINANT-style balanced eval: positives + same number of random negatives."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    aps, f1s = [], []
    for _ in range(n_trials):
        if len(neg_idx) <= len(pos_idx):
            sn = neg_idx
        else:
            sn = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        keep = np.concatenate([pos_idx, sn])
        sub_y = y_true[keep]
        sub_s = y_score[keep]
        try:
            aps.append(average_precision_score(sub_y, sub_s))
        except ValueError:
            aps.append(0.0)
        ths = np.linspace(sub_s.min(), sub_s.max(), 100)
        bf1 = 0.0
        for t in ths:
            yp = (sub_s >= t).astype(int)
            f = f1_score(sub_y, yp, average='macro', zero_division=0)
            if f > bf1:
                bf1 = f
        f1s.append(bf1)
    return float(np.mean(aps)), float(np.mean(f1s))


def compute_all_metrics(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if y_score.max() > y_score.min():
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    else:
        y_score_norm = y_score

    auroc = compute_auroc(y_true, y_score_norm)
    auprc = compute_auprc(y_true, y_score_norm)
    macro_f1 = compute_macro_f1(y_true, y_score_norm)
    auprc_bal, macro_f1_bal = compute_balanced_auprc_f1(y_true, y_score_norm)

    num_anomalies = int(np.sum(y_true))
    k_values = [50, 100, 200, num_anomalies] if num_anomalies > 0 else [50, 100, 200]

    precision_at_k = {}
    recall_at_k = {}
    for k in k_values:
        if k <= len(y_true):
            precision_at_k[k] = compute_precision_at_k(y_true, y_score_norm, k)
            recall_at_k[k] = compute_recall_at_k(y_true, y_score_norm, k)

    return {
        'auroc': auroc,
        'auprc': auprc,
        'auprc_bal': auprc_bal,
        'macro_f1': macro_f1,
        'macro_f1_bal': macro_f1_bal,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'num_anomalies': num_anomalies,
        'num_samples': len(y_true)
    }

def aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}
    
    aggregated = {}
    keys = ['auroc', 'auprc', 'auprc_bal', 'macro_f1', 'macro_f1_bal']
    
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return aggregated

def format_metric(mean, std, precision=1):
    return f"{mean*100:.{precision}f}±{std*100:.{precision}f}"

def format_all_metrics(aggregated_metrics, precision=1):
    formatted = {}
    for key, stats in aggregated_metrics.items():
        formatted[key] = format_metric(stats['mean'], stats['std'], precision)
    return formatted
