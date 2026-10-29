import os
import json
import pickle
import torch
import numpy as np

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def _json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath, map_location='cpu'):
    return torch.load(filepath, map_location=map_location)

def save_pickle(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
