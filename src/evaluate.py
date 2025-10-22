import numpy as np
from typing import Dict, Any, Tuple
from train import decision_tree_learning

def predict(sample: np.ndarray, tree: Dict[str, Any]) -> float:
    """Recursively predicts the label for a single sample."""
    
    # Base case: if we are at a leaf node, return its label
    if tree['type'] == 'leaf':
        return tree['label']
    
    # Recursive step: traverse the tree
    attribute = tree['attr']
    threshold = tree['thr']
    
    if sample[attribute] <= threshold:
        return predict(sample, tree['left'])
    else:
        return predict(sample, tree['right'])

def evaluate(test_db: np.ndarray, trained_tree: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the accuracy of a trained tree on a test dataset.
    
    Returns:
        (accuracy, predictions, true_labels)
    """
    true_labels = test_db[:, -1]
    features = test_db[:, :-1]
    
    predictions = []
    for i in range(len(features)):
        sample = features[i]
        pred = predict(sample, trained_tree)
        predictions.append(pred)
        
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == true_labels)
    
    return accuracy, predictions, true_labels

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Computes the confusion matrix."""
    n_labels = len(labels)
    # Create a mapping from label value (e.g., 1.0, 2.0) to matrix index (0, 1)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_index[true]
        pred_idx = label_to_index[pred]
        matrix[true_idx, pred_idx] += 1
        
    return matrix

def calculate_metrics(conf_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates accuracy, precision, recall, and F1-score from a confusion matrix."""
    n_classes = conf_matrix.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        
        # Add 1e-9 to avoid division by zero
        precision[i] = TP / (TP + FP + 1e-9)
        recall[i] = TP / (TP + FN + 1e-9)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-9)
        
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    return accuracy, precision, recall, f1

def cross_validation(data: np.ndarray, k: int = 10) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs k-fold cross-validation.
    
    Returns:
        (avg_conf_matrix, accuracy, precision, recall, f1, labels)
    """
    # Shuffle the dataset
    np.random.seed(42) # for reproducibility
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data)
    
    # Split data into k folds
    folds = np.array_split(shuffled_data, k)
    
    all_preds = []
    all_true = []
    
    print(f"Starting {k}-fold cross-validation...")
    for i in range(k):
        print(f"  Fold {i+1}/{k}")
        test_set = folds[i]
        train_list = [folds[j] for j in range(k) if j != i]
        train_set = np.concatenate(train_list, axis=0)
        
        # Train the model
        trained_tree, _ = decision_tree_learning(train_set, 0)
        
        # Evaluate on the test set
        _, fold_preds, fold_true = evaluate(test_set, trained_tree)
        
        all_preds.extend(fold_preds)
        all_true.extend(fold_true)
        
    print("Cross-validation complete.")
        
    # Get unique labels (e.g., [1., 2., 3., 4.])
    labels = np.unique(data[:, -1])
    
    # Compute metrics based on all (out-of-sample) predictions
    conf_matrix = confusion_matrix(np.array(all_true), np.array(all_preds), labels)
    accuracy, precision, recall, f1 = calculate_metrics(conf_matrix)
    
    return conf_matrix, accuracy, precision, recall, f1, labels
