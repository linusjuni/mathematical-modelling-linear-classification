import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_weights(model, image_shape=(224, 224)):
    # Extract model weights (excluding bias)
    weights = model.coef_[0]
    
    weight_image = weights.reshape(image_shape)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(weight_image, cmap='seismic')
    plt.colorbar()
    plt.title('Model Weights (Blue = Negative, Red = Positive)')
    plt.show()

def visualize_lambda_selection(cv_df):
    if 'lambda' not in cv_df.columns:
        raise ValueError("cv_df must contain a 'lambda' column.")

    plt.figure(figsize=(10, 6))
    lambda_counts = cv_df['lambda'].value_counts().sort_index()
    lambda_labels = lambda_counts.index.astype(str)
    counts = lambda_counts.values

    plt.bar(lambda_labels, counts)
    plt.xlabel('Lambda Value')
    plt.ylabel('Selection Frequency')
    plt.title('Lambda Selection Frequency in Cross-Validation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_performance_by_lambda(cv_df, metric):
    """
    Visualizes the estimated generalization error (outer fold performance)
    for the lambdas selected during nested cross-validation.
    """
    if metric not in cv_df.columns:
        print(f"Warning: Metric '{metric}' not found in cv_results. Available metrics: {cv_df.columns.tolist()}")
        return

    # Group by selected lambda and calculate mean/std of the chosen metric
    performance_by_lambda = cv_df.groupby('lambda')[metric].agg(['mean', 'std']).reset_index()
    performance_by_lambda = performance_by_lambda.sort_values('lambda') # Ensure sorted for plotting

    plt.figure(figsize=(10, 6))
    plt.errorbar(performance_by_lambda['lambda'], performance_by_lambda['mean'],
                 yerr=performance_by_lambda['std'], fmt='-o', capsize=5,
                 label=f'Mean Outer Fold {metric.upper()} ± Std Dev')

    plt.xscale('log') # Lambdas often span orders of magnitude
    plt.xlabel('Selected Lambda (log scale)')
    plt.ylabel(f'Outer Fold {metric.upper()}')
    plt.title(f'Generalization Performance vs. Selected Lambda ({metric.upper()})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    # Consider saving the plot instead of just showing it
    # plt.savefig(os.path.join(base_path, "plots", "generalization_error_vs_lambda.png"))
    plt.show()  