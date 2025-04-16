from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_models(X_train, y_train, lambda_values):
    models = {}
    for lambda_val in lambda_values:
        print(f"Training model with lambda: {lambda_val}...")
        model = LogisticRegression(C=1/lambda_val, penalty='l2', solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        models[lambda_val] = model
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for lambda_val, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[lambda_val] = accuracy
    return results