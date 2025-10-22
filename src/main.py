import numpy as np
from visual import visualize_tree
from train import decision_tree_learning
from evaluate import cross_validation, calculate_metrics

NUM_SIGNALS = 7

def load_data_from_file(file_name: str) -> np.ndarray:
    try:
        return np.loadtxt(file_name)
    except IOError as e:
        print(f"Error loading file {file_name}: {e}")
        print("Please ensure 'WIFI_db/clean_dataset.txt' and 'WIFI_db/noisy_dataset.txt' exist.")
        exit(1)

def print_metrics(conf_matrix, accuracy, precision, recall, f1, labels, title):
    
    print(f"\n===== {title} =====")
    
    print("\n## Confusion Matrix")
    header = "True \\ Pred |"
    for label in labels:
        header += f" Room {int(label)} |"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(conf_matrix):
        row_str = f" Room {int(labels[i])}    |"
        for val in row:
            row_str += f" {val:<7} |"
        print(row_str)

    print("\n## Overall Accuracy")
    print(f"{accuracy * 100:.2f}%")

    print("\n## Per-Class Metrics")
    print(f"{'Class (Room)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 57)
    for i, label in enumerate(labels):
        print(f"Room {int(label):<9} | {precision[i]:<10.3f} | {recall[i]:<10.3f} | {f1[i]:<10.3f}")

if __name__ == '__main__':
    clean_data = load_data_from_file("src/For_60012/wifi_db/clean_dataset.txt")
    noisy_data = load_data_from_file("src/For_60012/wifi_db/noisy_dataset.txt")

    clean_tree, clean_depth = decision_tree_learning(clean_data, 0)
    
    visualize_tree(clean_tree, clean_depth, "decision_tree_visualization.png")

    conf_matrix_clean, labels = cross_validation(clean_data, k=10)
    accuracy, precision, recall, f1 = calculate_metrics(conf_matrix_clean)
    print_metrics(conf_matrix_clean, accuracy, precision, recall, f1, labels, title="10-Fold CV Metrics: Clean Dataset")

    conf_matrix_noisy, labels = cross_validation(noisy_data, k=10)
    accuracy, precision, recall, f1 = calculate_metrics(conf_matrix_noisy)
    print_metrics(conf_matrix_noisy, accuracy, precision, recall, f1, labels, title="10-Fold CV Metrics: Noisy Dataset")










