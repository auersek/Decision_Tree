import numpy as np
import matplotlib
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any
from load import load_data_from_file
from split import entropy, find_split

NUM_SIGNALS = 7

def load_data_from_file(file_name: str) -> np.ndarray:
    """
    Loads the dataset from a file.
    """
    try:
        return np.loadtxt(file_name)
    except IOError as e:
        print(f"Error loading file {file_name}: {e}")
        print("Please ensure 'WIFI_db/clean_dataset.txt' and 'WIFI_db/noisy_dataset.txt' exist.")
        exit(1)

def print_metrics(conf_matrix, accuracy, precision, recall, f1, labels, title):
    """Helper function to print all metrics in a formatted way."""
    
    print(f"\n===== {title} =====")
    
    # --- Confusion Matrix ---
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

    # --- Overall Accuracy ---
    print("\n## Overall Accuracy")
    print(f"{accuracy * 100:.2f}%")

    # --- Per-Class Metrics ---
    print("\n## Per-Class Metrics")
    print(f"{'Class (Room)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 57)
    for i, label in enumerate(labels):
        print(f"Room {int(label):<9} | {precision[i]:<10.3f} | {recall[i]:<10.3f} | {f1[i]:<10.3f}")

# --- Main execution ---

if __name__ == '__main__':

    # --- Step 1: Load Data ---
    clean_data = load_data_from_file("WIFI_db/clean_dataset.txt")
    noisy_data = load_data_from_file("WIFI_db/noisy_dataset.txt")
    print("Data loaded successfully.")

    # --- Step 2 (Report): Train on full clean dataset for visualization ---
    print("\nTraining tree on full clean dataset for visualization...")
    clean_tree, clean_depth = decision_tree_learning(clean_data, 0)
    print(f"Tree trained. Max depth: {clean_depth}")
    
    # Generate and save the plot
    visualize_tree(clean_tree, clean_depth, "decision_tree_visualization.png")

    # --- Step 3 (Report): Evaluation ---

    # Run 10-fold CV on Clean Dataset
    metrics_clean = cross_validation(clean_data, k=10)
    print_metrics(*metrics_clean, title="10-Fold CV Metrics: Clean Dataset")
    
    # Run 10-fold CV on Noisy Dataset
    metrics_noisy = cross_validation(noisy_data, k=10)
    print_metrics(*metrics_noisy, title="10-Fold CV Metrics: Noisy Dataset")










