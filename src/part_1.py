import numpy as np

from data_utils import load_data
from model_utils import train_models, evaluate_models
from visualization_utils import visualize_weights

train_path = "/Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data/test"
test_path = "/Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data/train"

print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

lambda_values = np.concatenate([
    np.logspace(-5, -3, 5),
    np.linspace(0.001, 0.1, 10), 
    np.linspace(0.1, 1, 5),
    np.linspace(1, 10, 5),
    np.linspace(10, 1000, 5)
])
lambda_values = np.unique(lambda_values.round(8))

print("Training models...")
models = train_models(X_train, y_train, lambda_values)

print("Evaluating models...")
results = evaluate_models(models, X_test, y_test)

best_lambda = max(results, key=results.get)
best_model = models[best_lambda]
best_accuracy = results[best_lambda]

print(f"Best regularization strength (lambda): {best_lambda}")
print(f"Best model accuracy: {best_accuracy:.4f}")

print("\nAccuracy for different regularization strengths:")
for lambda_val, accuracy in sorted(results.items()):
    print(f"lambda = {lambda_val}: {accuracy:.4f}")

print("Visualizing weights of best model...")
visualize_weights(best_model)