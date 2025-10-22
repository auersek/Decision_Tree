import numpy as np
import matplotlib

# class Nodes:
    


def load_data_from_file(file_name):
    data = []
    with open(file_name, "r") as file:
        for line in file:
            row = list(map(int, line.split()))
            data.append(row)
    print(data)
    return data

def decision_tree_learning(mat_A, depth):
    return 0

def evaluate(test_db, trained_tree):
    return 0


if __name__ == '__main__':

    wifi_dict = {}

    load_data_from_file("src/For_60012/wifi_db/clean_dataset.txt")









