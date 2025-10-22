def evaluate(test_db, trained_tree):
    x_coord = test_db[:, :-1]
    y_true = test_db[:, -1]
    y_pred = np.array([predict(trained_tree, sample) for sample in X])

    accuracy = np.mean(y_pred == y_true)
    return 0

def predict(trained_tree, sample):
    if trained_tree["type"] == "leaf":
        return trained_tree["label"]
    
    attr = trained_tree["attr"]
    thr = trained_tree["thr"]


    if sample[attr] <= thr:

        return predict(trained_tree["left"], sample)
    else:
        return predict(trained_tree["right"], sample)