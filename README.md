# Decision Tree Coursework
The program will:
1.  Train a tree on the full clean dataset and save it as `decision_tree_visualization.png`.
2.  Run a nested 10-fold cross-validation on the **clean dataset** and print all metrics
3.  Run a nested 10-fold cross-validation on the **noisy dataset (default), or provided dataset** and print all metrics

## 1. Setup

### Dependencies

This project requires Python 3 and the following libraries:
* `numpy`
* `matplotlib`

### Installation

Follow the following steps to get setup:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/auersek/Decision_Tree
    cd Decision_Tree
    ```

2.  **Install libraries:**
    ```bash
    pip install numpy matplotlib
    ```

## 2. Data

The script requires the provided datasets to be in a specific location. `clean_dataset.txt` and `noisy_dataset.txt` are already inside the `wifi_db` folder. Move your txt input data into the same folder.

## 3. How to Run

The script takes in an argument for the file path to your input data. If this is not provided, the tree will run on the provided noisy dataset.

```bash
python3 src/main.py --file-name src/wifi_db/{your_filename}
```
