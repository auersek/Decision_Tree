from collections import Counter
import numpy as np
from typing import Dict, Any, Tuple

def entropy(labels: np.ndarray) -> float:
    
    n = len(labels)
    if n == 0:
        return 0.0
    
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    
    return -np.sum(probabilities * np.log2(probabilities))

def find_split(dataset: np.ndarray) -> Tuple[int, float]:

    n_samples, n_features_total = dataset.shape
    n_features = n_features_total - 1 
    
    labels = dataset[:, -1]
    
    if len(np.unique(labels)) == 1:
        return -1, float("nan")

    parent_entropy = entropy(labels)
    best_gain = -1.0
    best_attr = -1
    best_thr = float("nan")

    for j in range(n_features):
        sorted_indices = np.argsort(dataset[:, j])
        feature_col = dataset[sorted_indices, j]
        sorted_labels = labels[sorted_indices]

        for k in range(n_samples - 1):
            if feature_col[k] == feature_col[k+1]:
                continue
            
            thr = (feature_col[k] + feature_col[k+1]) / 2.0
            
            left_labels = sorted_labels[:k+1]
            right_labels = sorted_labels[k+1:]

            remainder = (len(left_labels) / n_samples) * entropy(left_labels) + \
                          (len(right_labels) / n_samples) * entropy(right_labels)
            
            gain = parent_entropy - remainder
            
            if gain > best_gain:
                best_gain = gain
                best_attr = j
                best_thr = thr
                
    if best_gain <= 0:
        return -1, float("nan")

    return best_attr, best_thr

def split_dataset(dataset: np.ndarray, attr: int, thr: float) -> Tuple[np.ndarray, np.ndarray]:

    left_mask = dataset[:, attr] <= thr
    right_mask = dataset[:, attr] > thr
    return dataset[left_mask], dataset[right_mask]

def all_same_label(dataset: np.ndarray) -> bool:

    if len(dataset) == 0:
        return True
    labels = dataset[:, -1]
    return len(np.unique(labels)) == 1

def majority_label(labels: np.ndarray) -> float:

    return Counter(labels).most_common(1)[0][0]

def decision_tree_learning(train_data: np.ndarray, depth: int) -> Tuple[Dict[str, Any], int]:
    
    if all_same_label(train_data):

        label = train_data[0, -1] if len(train_data) > 0 else 1
        return {"type": "leaf", "label": label}, depth
    
    attr, thr = find_split(train_data)
    
    if attr == -1:
        labels = train_data[:, -1]
        return {"type": "leaf", "label": majority_label(labels)}, depth

    node = {
        "type": "node",
        "attr": attr,
        "thr": thr,
    }
    
    l_dataset = train_data[train_data[:, attr] <= thr]
    r_dataset = train_data[train_data[:, attr] > thr]
    
    if len(l_dataset) == 0 or len(r_dataset) == 0:
        labels = train_data[:, -1]
        return {"type": "leaf", "label": majority_label(labels)}, depth

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    
    node["left"] = l_branch
    node["right"] = r_branch

    return node, max(l_depth, r_depth)
