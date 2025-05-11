import os
import numpy as np
import pandas as pd

from data_utils import load_data, load_data_with_histograms_of_orientation, load_data_with_sobel_kernel
from visualization_utils import plot_pca_2d, plot_pca_3d, plot_pca_scree, plot_combined_results

base_path = "/Users/linusjuni/Documents/General Engineering/6. Semester/Mathematical Modelling/Assignments/mathematical-modelling-linear-classification/"
data_path = os.path.join(base_path, "data")

train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)
print("Combing both datasets")
X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

print("Plotting PCA 2D...")
plot_pca_2d(X_train, y_train)
print("Plotting PCA 3D...")
plot_pca_3d(X_train, y_train)
print("Plotting PCA Scree...")
plot_pca_scree(X_train)