from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
import joblib

def train_model(X, y, lambda_val):
    model = LogisticRegression(C=1/lambda_val, penalty='l2', solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def save_model(model, filepath):
    """Saves the trained model to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Loads a trained model from a file."""
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")
        return None
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def find_best_lambda(X_train, y_train, lambda_values, n_inner_folds):
    """
    Use inner cross-validation to find the best lambda value
    """

    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=69)
    
    best_lambda = None
    best_score = -np.inf
    
    for lambda_val in lambda_values:
        print(f"    Evaluating lambda: {lambda_val}")
        inner_scores = []
        
        # Evaluate this lambda value using inner CV
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            print(f"        Inner Fold {len(inner_scores)+1}/{n_inner_folds}")
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            
            model = train_model(X_inner_train, y_inner_train, lambda_val)
            
            y_val_pred_proba = model.predict_proba(X_inner_val)[:, 1]
            score = roc_auc_score(y_inner_val, y_val_pred_proba)
            inner_scores.append(score)
        
        mean_inner_score = np.mean(inner_scores)
        
        if mean_inner_score > best_score:
            best_score = mean_inner_score
            best_lambda = lambda_val
    
    return best_lambda, best_score

def nested_cross_validation(X, y, lambda_values, n_inner, n_outer):
    """
    Performs nested cross-validation to select the best lambda and evaluate model performance.
    """

    # Initialize outer cross-validation splitter
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=69)
    
    outer_results = []
    selected_lambdas = []
    
    # Outer loop - estimate generalization performance
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer Fold {fold_idx+1}/{n_outer}")
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        best_lambda, best_score = find_best_lambda(X_train_outer, y_train_outer, lambda_values, n_inner)
        print(f"\n    Selected lambda: {best_lambda} (inner CV score: {best_score:.4f})")
        
        model = train_model(X_train_outer, y_train_outer, best_lambda)
        
        test_metrics = evaluate_model(model, X_test_outer, y_test_outer)
        
        outer_results.append({
            'fold': fold_idx,
            'lambda': best_lambda,
            'inner_score': best_score,
            'accuracy': test_metrics['accuracy'],
            'auc': test_metrics['auc']
        })
        
        selected_lambdas.append(best_lambda)
        
        print(f"    Outer test accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}\n")
    
    best_lambda = max(set(selected_lambdas), key=selected_lambdas.count)
    print(f"Out of the selected lambdas {[int(lambda_val) for lambda_val in selected_lambdas]}, the most frequent is {best_lambda}")

    final_model = train_model(X, y, best_lambda)
    
    return best_lambda, final_model, outer_results