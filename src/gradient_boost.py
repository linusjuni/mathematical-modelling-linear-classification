import os
from sklearn.ensemble import HistGradientBoostingClassifier # Use this for speed
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

gb_clf = HistGradientBoostingClassifier(max_iter=50, max_depth=3, learning_rate=0.1, verbose=1) 
print("Training Gradient Boosting classifier...")
gb_clf.fit(X_train, y_train)

print("Evaluating Gradient Boosting classifier...")
y_pred = gb_clf.predict(X_test)
y_pred_proba = gb_clf.predict_proba(X_test)[:, 1] # Probabilities for the positive class

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")