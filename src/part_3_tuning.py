import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from data_utils import load_data_with_histograms_of_orientation
from model_utils import nested_cross_validation, train_model, evaluate_model, save_model, load_model
from visualization_utils import visualize_weights, visualize_lambda_selection, visualize_performance_by_lambda

def tune_hog_parameters(base_path, n_components_pca=None, visualize=True):
    pixels_per_cell_options = [(8, 8), (16, 16), (32, 32), (64, 64)]
    orientations_options = [9, 18, 36, 72]
    cells_per_block_options = [(1, 1), (2, 2), (3, 3)]
    
    results = []
    
    data_path = os.path.join(base_path, "data")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    
    lambda_values = np.concatenate([
        np.logspace(-5, -3, 3),
        np.linspace(0.01, 1, 5),
        np.linspace(10, 100, 3)
    ])
    lambda_values = np.unique(lambda_values.round(8))
    
    N_INNER = 5
    N_OUTER = 3
    
    total_combinations = len(pixels_per_cell_options) * len(orientations_options) * len(cells_per_block_options)
    counter = 0
    
    for pixels_per_cell in pixels_per_cell_options:
        for orientations in orientations_options:
            for cells_per_block in cells_per_block_options:
                counter += 1
                print(f"\nTesting combination {counter}/{total_combinations}:")
                print(f"  pixels_per_cell={pixels_per_cell}, orientations={orientations}, cells_per_block={cells_per_block}")
                
                X_train, y_train = load_data_with_histograms_of_orientation(
                    train_path, 
                    pixels_per_cell=pixels_per_cell,
                    orientations=orientations,
                    cells_per_block=cells_per_block
                )
                
                X_test, y_test = load_data_with_histograms_of_orientation(
                    test_path,
                    pixels_per_cell=pixels_per_cell,
                    orientations=orientations,
                    cells_per_block=cells_per_block
                )
                
                if n_components_pca is not None:
                    pca = PCA(n_components=n_components_pca)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                
                best_lambda, best_model, cv_results = nested_cross_validation(
                    X_train, y_train, lambda_values, n_inner=N_INNER, n_outer=N_OUTER
                )
                
                test_metrics = evaluate_model(best_model, X_test, y_test)
                
                cv_df = pd.DataFrame(cv_results)
                result = {
                    'pixels_per_cell': f"{pixels_per_cell[0]}x{pixels_per_cell[1]}",
                    'orientations': orientations,
                    'cells_per_block': f"{cells_per_block[0]}x{cells_per_block[1]}",
                    'best_lambda': best_lambda,
                    'cv_accuracy_mean': cv_df['accuracy'].mean(),
                    'cv_accuracy_std': cv_df['accuracy'].std(),
                    'cv_auc_mean': cv_df['auc'].mean(),
                    'cv_auc_std': cv_df['auc'].std(),
                    'test_accuracy': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'],
                    'feature_dimension': X_train.shape[1]
                }
                
                results.append(result)
                print(f"  CV AUC: {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}")
                print(f"  Test AUC: {result['test_auc']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['test_auc'].idxmax()
    best_params = results_df.iloc[best_idx]
    
    print("\nBest HOG parameters:")
    print(f"  pixels_per_cell={best_params['pixels_per_cell']}")
    print(f"  orientations={best_params['orientations']}")
    print(f"  cells_per_block={best_params['cells_per_block']}")
    print(f"  Test AUC: {best_params['test_auc']:.4f}")
    
    if visualize:
        visualize_hog_tuning_results(results_df)
    
    return best_params, results_df

def visualize_hog_tuning_results(results_df):
    
    plt.figure(figsize=(14, 5))
    param_labels = [f"{row['pixels_per_cell']}\n{row['orientations']}\n{row['cells_per_block']}" 
                   for _, row in results_df.iterrows()]
    indices = np.arange(len(results_df))
    bars = plt.bar(indices, results_df['test_auc'])
    plt.xticks(indices, param_labels, rotation=90, fontsize=8)
    plt.xlabel('HOG Parameters\n(pixels_per_cell, orientations, cells_per_block)')
    plt.ylabel('Test AUC')
    plt.title('Performance by HOG Parameter Combination')
    
    best_idx = results_df['test_auc'].idxmax()
    bars[best_idx].set_color('red')
    plt.tight_layout()
    plt.savefig('hog_parameter_combinations.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['feature_dimension'], results_df['test_auc'])
    plt.xlabel('Feature Dimension')
    plt.ylabel('Test AUC')
    plt.title('Performance vs Feature Dimension')
    plt.tight_layout()
    plt.savefig('hog_feature_dimension.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    results_df.boxplot(column='test_auc', by='pixels_per_cell', grid=False)
    plt.title('Performance by Pixels Per Cell')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('hog_pixels_per_cell.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    results_df.boxplot(column='test_auc', by='orientations', grid=False)
    plt.title('Performance by Orientations')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('hog_orientations.png')
    plt.show()

def run_part3_tuning(n_components_pca=None, visualize=False, tune_hog=False):

    base_path = "/Users/linusjuni/Documents/General Engineering/6. Semester/Mathematical Modelling/Assignments/mathematical-modelling-linear-classification/"
    
    if tune_hog:
        print("Starting HOG parameter tuning...")
        best_params, results_df = tune_hog_parameters(base_path, n_components_pca, visualize=True)
        
        pixels_parts = best_params['pixels_per_cell'].split('x')
        pixels_per_cell = (int(pixels_parts[0]), int(pixels_parts[1]))
        
        cells_parts = best_params['cells_per_block'].split('x')
        cells_per_block = (int(cells_parts[0]), int(cells_parts[1]))
        
        orientations = best_params['orientations']
        
        print("\nRunning with best HOG parameters...")
        return run_part3_with_params(base_path, pixels_per_cell, orientations, cells_per_block, 
                                   n_components_pca, visualize)
    else:
        pixels_per_cell = (32, 32)
        orientations = 36
        cells_per_block = (1, 1)
        return run_part3_with_params(base_path, pixels_per_cell, orientations, cells_per_block, 
                                   n_components_pca, visualize)

def run_part3_with_params(base_path, pixels_per_cell, orientations, cells_per_block, 
                        n_components_pca=None, visualize=False):
    N_INNER = 10
    N_OUTER = 10

    data_path = os.path.join(base_path, "data")
    model_suffix = "_pca" if n_components_pca is not None else ""
    model_save_path = os.path.join(base_path, "models", 
                                 f"part_3_logistic_regression_model{model_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib")

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    
    print("Loading training data...")
    print(f"HOG parameters: pixels_per_cell={pixels_per_cell}, orientations={orientations}, cells_per_block={cells_per_block}")
    X_train, y_train = load_data_with_histograms_of_orientation(
        train_path, 
        pixels_per_cell=pixels_per_cell,
        orientations=orientations,
        cells_per_block=cells_per_block
    )
    
    print("Loading test data...")
    X_test, y_test = load_data_with_histograms_of_orientation(
        test_path, 
        pixels_per_cell=pixels_per_cell,
        orientations=orientations,
        cells_per_block=cells_per_block
    )

    if n_components_pca is not None:
        print(f"Reducing dimensionality using PCA to {n_components_pca} components...")
        pca = PCA(n_components=n_components_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    else:
        print("No PCA applied, using original data dimensions.")

    lambda_values = np.concatenate([
        np.logspace(-5, -3, 5),
        np.linspace(0.001, 0.1, 10), 
        np.linspace(0.1, 1, 5),
        np.linspace(1, 10, 5),
        np.linspace(10, 1000, 5)
    ])
    lambda_values = np.unique(lambda_values.round(8))

    print(f"Running nested cross-validation with {len(lambda_values)} lambda values...")
    best_lambda, best_model, cv_results = nested_cross_validation(X_train, y_train, lambda_values, n_inner=N_INNER, n_outer=N_OUTER)

    cv_df = pd.DataFrame(cv_results)
    print("\nCross-validation results:")
    print(f"Mean accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"Mean AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    print(f"Selected lambda values across folds: {cv_df['lambda'].tolist()}")
    print(f"Best lambda (most frequently selected): {best_lambda}")

    test_metrics = evaluate_model(best_model, X_test, y_test)
    print("\nFinal model performance on test set:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")

    print(f"\nSaving the best model trained on all data with lambda={best_lambda}...")
    save_model(best_model, model_save_path)

    print(f"Parameters used for HOG:")
    print(f"  pixels_per_cell: {pixels_per_cell}")
    print(f"  orientations: {orientations}")
    print(f"  cells_per_block: {cells_per_block}")

    if visualize:
        print("Visualizing weights of best model...")
        visualize_weights(best_model)

        print("Visualizing lambda selection frequency...")
        visualize_lambda_selection(cv_df)
        
        print("Visualizing generalization error vs selected lambda...")
        visualize_performance_by_lambda(cv_df, metric='auc')

    return {
        'best_lambda': best_lambda,
        'cv_df': cv_df,
        'cv_accuracy_list':  cv_df['accuracy'].tolist(),
        'cv_auc_list': cv_df['auc'].tolist(),
        'cv_accuracy_mean': cv_df['accuracy'].mean(),
        'cv_accuracy_std': cv_df['accuracy'].std(),
        'cv_auc_mean': cv_df['auc'].mean(),
        'cv_auc_std': cv_df['auc'].std(),
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
        'y_test': y_test,
        'y_test_pred': test_metrics['y_pred'],
        'y_test_prob': test_metrics['y_prob'],
        'model': best_model,
        'hog_params': {
            'pixels_per_cell': pixels_per_cell,
            'orientations': orientations,
            'cells_per_block': cells_per_block
        }
    }

if __name__ == "__main__":
    run_part3_tuning(n_components_pca=0.95, visualize=False, tune_hog=True)