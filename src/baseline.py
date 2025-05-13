import numpy as np
import os
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from data_utils import load_data

def run_baseline_model():
    base_path = "/Users/linusjuni/Documents/General Engineering/6. Semester/Mathematical Modelling/Assignments/mathematical-modelling-linear-classification/"
    data_path = os.path.join(base_path, "data")

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    print("Loading training data for baseline...")
    _, y_train = load_data(train_path)
    
    print("Loading test data for baseline...")
    _, y_test = load_data(test_path)

    # Determine the majority class in the training set
    num_positive_train = np.sum(y_train)
    num_negative_train = len(y_train) - num_positive_train

    if num_positive_train >= num_negative_train:
        majority_class = 1
        print(f"Majority class in training set: Positive ({num_positive_train} samples)")
    else:
        majority_class = 0
        print(f"Majority class in training set: Negative ({num_negative_train} samples)")

    y_pred_baseline = np.full_like(y_test, majority_class)

    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
    
    # For AUC, predict_proba would be [0, 1] or [1, 0] depending on the majority class
    # If majority_class is 1, prob of positive is 1.0 for all samples.
    # If majority_class is 0, prob of positive is 0.0 for all samples.
    y_prob_baseline = np.zeros((len(y_test), 2))
    if majority_class == 1:
        y_prob_baseline[:, 1] = 1.0 # Probability of positive class
    else:
        y_prob_baseline[:, 0] = 1.0 # Probability of negative class
    
    auc_baseline = roc_auc_score(y_test, y_prob_baseline[:, 1])

    print("\nBaseline Model Performance on Test Set:")
    print(f"Accuracy: {accuracy_baseline:.4f}")
    print(f"AUC: {auc_baseline:.4f}")

    return {
        'majority_class': majority_class,
        'test_accuracy': accuracy_baseline,
        'test_auc': auc_baseline,
        'y_test': y_test,
        'y_pred_baseline': y_pred_baseline
    }

if __name__ == "__main__":
    run_baseline_model()
