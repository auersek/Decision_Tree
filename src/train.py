from collections import Counter
import numpy as np
from typing import Dict, Any, Tuple

def entropy(labels: np.ndarray) -> float:
    """
    Calculates the Shannon entropy of a NumPy array of labels.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    
    # np.unique(..., return_counts=True) is the NumPy equivalent of Counter
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    
    return -np.sum(probabilities * np.log2(probabilities))

def find_split(dataset: np.ndarray) -> Tuple[int, float]:
    """
    Finds the best attribute and threshold for a split, maximizing info gain.
    
    Returns:
        (best_attribute_index, best_threshold)
        Returns (-1, float("nan")) if no valid split is found.
    """
    n_samples, n_features_total = dataset.shape
    # Last column is label, so 7 features (0-6)
    n_features = n_features_total - 1 
    
    labels = dataset[:, -1]
    
    # Base case: cannot split if all labels are the same or not enough samples
    if len(np.unique(labels)) == 1:
        return -1, float("nan")

    parent_entropy = entropy(labels)
    best_gain = -1.0
    best_attr = -1
    best_thr = float("nan")

    for j in range(n_features):
        # Sort data by the current feature
        # Using argsort is efficient as it just gives indices
        sorted_indices = np.argsort(dataset[:, j])
        feature_col = dataset[sorted_indices, j]
        sorted_labels = labels[sorted_indices]

        # Iterate through possible split points
        for k in range(n_samples - 1):
            # Only consider splits between different values
            if feature_col[k] == feature_col[k+1]:
                continue
            
            # Define the split point
            thr = (feature_col[k] + feature_col[k+1]) / 2.0
            
            # Split the labels
            left_labels = sorted_labels[:k+1]
            right_labels = sorted_labels[k+1:]

            # Calculate remainder (weighted average of children entropy)
            nL, nR = len(left_labels), len(right_labels)
            remainder = (nL / n_samples) * entropy(left_labels) + \
                          (nR / n_samples) * entropy(right_labels)
            
            # Calculate information gain
            gain = parent_entropy - remainder
            
            if gain > best_gain:
                best_gain = gain
                best_attr = j
                best_thr = thr
                
    # Check if any positive gain was found
    if best_gain <= 0:
        return -1, float("nan")

    return best_attr, best_thr

def split_dataset(dataset: np.ndarray, attr: int, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset into left and right subsets based on a threshold.
    This uses efficient NumPy boolean masking.
    """
    left_mask = dataset[:, attr] <= thr
    right_mask = dataset[:, attr] > thr
    return dataset[left_mask], dataset[right_mask]

def all_same_label(dataset: np.ndarray) -> bool:
    """Checks if all samples in the dataset have the same label."""
    if len(dataset) == 0:
        return True
    labels = dataset[:, -1]
    return len(np.unique(labels)) == 1

def majority_label(labels: np.ndarray) -> float:
    """Finds the most common label in a NumPy array."""
    # Counter works perfectly on NumPy arrays
    return Counter(labels).most_common(1)[0][0]

def decision_tree_learning(train_data: np.ndarray, depth: int) -> Tuple[Dict[str, Any], int]:
    """
    Recursively builds a decision tree as per the coursework pseudo-code.
    
    Returns:
        (node, max_depth)
        node: A dictionary representing the tree/subtree
        max_depth: The maximum depth reached from this node
    """
    
    # 1. Check if all samples have the same label (Base Case)
    if all_same_label(train_data):
        # Handle empty dataset by returning a default label (e.g., 1)
        label = train_data[0, -1] if len(train_data) > 0 else 1
        return {"type": "leaf", "label": label}, depth
    
    # 2. Find the best split
    attr, thr = find_split(train_data)
    
    # 3. No valid split found (e.g., no info gain, or all features identical)
    # This is an implied base case to prevent infinite loops.
    if attr == -1:
        labels = train_data[:, -1]
        return {"type": "leaf", "label": majority_label(labels)}, depth

    # 4. Create a new node
    node = {
        "type": "node",
        "attr": attr,
        "thr": thr,
    }
    
    # 5. Split training dataset
    l_dataset = train_data[train_data[:, attr] <= thr]
    r_dataset = train_data[train_data[:, attr] > thr]
    
    # 6. Handle edge case: if a split results in an empty child
    # This can happen if all feature values are identical but labels differ
    if len(l_dataset) == 0 or len(r_dataset) == 0:
        labels = train_data[:, -1]
        return {"type": "leaf", "label": majority_label(labels)}, depth
        
    # 7. Recursive calls
    # Note: We pass depth + 1, exactly as in the pseudo-code
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    
    node["left"] = l_branch
    node["right"] = r_branch
    
    # 8. Return the node and the max depth of its branches
    return node, max(l_depth, r_depth)
