import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from data_utils import load_data, load_data_with_sobel_kernel, load_data_with_histograms_of_orientation

base_path = "/Users/linusjuni/Documents/General Engineering/6. Semester/Mathematical Modelling/Assignments/mathematical-modelling-linear-classification/"
data_path = os.path.join(base_path, "data")

train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

log_reg_clf = LogisticRegression(max_iter=1000) # Added max_iter for convergence
print("Training Logistic Regression classifier...")
log_reg_clf.fit(X_train, y_train)

print("Making predictions on the test set...")
y_pred = log_reg_clf.predict(X_test)
y_pred_proba = log_reg_clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Classifier Performance:")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")