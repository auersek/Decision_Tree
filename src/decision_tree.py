import numpy as np
import matplotlib
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any

# class Nodes:


def load_data_from_file(file_name, train):
    return np.loadtxt(file_name)
    # data = []
    # validation_data = []
    # if train == "train":
    #     with open(file_name, "r") as file:
    #         for line in file:
    #             row = list(map(int, line.split()))
    #             data.append(row)
    # if train == "validation":       # unused for now but will be needed for later cross_validation
    #     with open(file_name, "r") as file:
    #         for line in file:
    #             row = list(map(int, line.split()))
    #             data.append(row)
            
    # return data

def entropy(labels: List[int]) -> float:
    """Shannon entropy of a list of class labels."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    from math import log2
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * log2(p)
    return H


def find_entropy_gain(feature_values: List[int], labels: List[int], threshold: float) -> float:
    """
    Information gain from splitting 'labels' by feature_values <= threshold vs > threshold.
    Matches your name 'find_entropy_gain'.
    """
    left_labels  = [y for x, y in zip(feature_values, labels) if x <= threshold]
    right_labels = [y for x, y in zip(feature_values, labels) if x > threshold]

    n = len(labels)
    nL, nR = len(left_labels), len(right_labels)
    if nL == 0 or nR == 0:
        return -1.0  # invalid split (no separation)

    base_H = entropy(labels)
    split_H = (nL / n) * entropy(left_labels) + (nR / n) * entropy(right_labels)
    return base_H - split_H


def find_split(dataset: List[List[int]]) -> Tuple[int, float, float]:
    """
    Search all 7 attributes and all midpoints between consecutive sorted values.
    Return (best_attribute_index, best_threshold, best_gain).
    If no useful split, returns (-1, None, 0.0)
    """
    if not dataset:
        return -1, float("nan"), 0.0

    # separate X and y
    X = [row[:7] for row in dataset]
    y = [row[7] for row in dataset]

    best_gain = -1.0
    best_attr = -1
    best_thr = float("nan")

    for j in range(7):
        # collect (value, label) and sort by feature value
        pairs = sorted((row[j], y_i) for row, y_i in zip(X, y))
        values = [v for v, _ in pairs]
        labels_sorted = [lab for _, lab in pairs]

        # candidate thresholds: midpoints between unique consecutive values
        for k in range(len(values) - 1):
            v1, v2 = values[k], values[k + 1]
            if v1 == v2:
                continue
            thr = (v1 + v2) / 2.0

            # compute gain for this feature and threshold
            gain = find_entropy_gain(values, labels_sorted, thr)
            if gain > best_gain:
                best_gain = gain
                best_attr = j
                best_thr = thr

    if best_gain < 0:
        return -1, float("nan"), 0.0
    return best_attr, best_thr, best_gain

def majority_label(labels: List[int]) -> int:
    return Counter(labels).most_common(1)[0][0]

def split_dataset(dataset: List[List[int]], attr: int, thr: float) -> Tuple[List[List[int]], List[List[int]]]:
    left, right = [], []
    for row in dataset:
        (left if row[attr] <= thr else right).append(row)
    return left, right

def all_same_label(dataset: List[List[int]]) -> bool:
    if not dataset:
        return True
    first = dataset[0][7]
    return all(row[7] == first for row in dataset)


def decision_tree_learning(training_dataset: List[List[int]], depth: int) -> Dict[str, Any]:
    """
    Build a simple binary decision tree:
    - Stop if depth==0, all labels equal, or no valid split -> return leaf with majority label.
    - Otherwise split on the best (attr, threshold).
    Tree node format:
      {'type':'leaf','label':int}
      {'type':'node','attr':int,'thr':float,'left':subtree,'right':subtree}
    """
    if not training_dataset:
        return {"type": "leaf", "label": 1}  # trivial fallback

    # stopping conditions
    if depth == 0 or all_same_label(training_dataset):
        return {"type": "leaf", "label": majority_label([r[7] for r in training_dataset])}

    attr, thr, gain = find_split(training_dataset)
    if attr == -1 or gain <= 0:
        return {"type": "leaf", "label": majority_label([r[7] for r in training_dataset])}

    left_ds, right_ds = split_dataset(training_dataset, attr, thr)
    if not left_ds or not right_ds:
        return {"type": "leaf", "label": majority_label([r[7] for r in training_dataset])}

    left_tree  = decision_tree_learning(left_ds,  depth - 1)
    right_tree = decision_tree_learning(right_ds, depth - 1)

    return {
        "type": "node",
        "attr": attr,
        "thr": thr,
        "gain": gain,
        "left": left_tree,
        "right": right_tree,
    }
    

    return 0

def evaluate(test_db, trained_tree):
    return 0


if __name__ == '__main__':

    wifi_dict = {}

    data = load_data_from_file("src/For_60012/wifi_db/clean_dataset.txt", "train")
    decision_tree_learning(data, )










