import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_weights(model, image_shape=(224, 224)):
    weights = model.coef_[0]
    
    weight_image = weights.reshape(image_shape)
    abs_weight_image = np.abs(weight_image)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original weights
    im0 = axes[0].imshow(weight_image, cmap='seismic')
    axes[0].set_title('Original Model Weights\n(Blue = Negative, Red = Positive)')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot absolute weights
    im1 = axes[1].imshow(abs_weight_image, cmap='viridis')
    axes[1].set_title('Absolute Model Weights')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
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
    if metric not in cv_df.columns:
        print(f"Warning: Metric '{metric}' not found in cv_results. Available metrics: {cv_df.columns.tolist()}")
        return

    performance_by_lambda = cv_df.groupby('lambda')[metric].agg(['mean', 'std']).reset_index()
    performance_by_lambda = performance_by_lambda.sort_values('lambda') # Ensure sorted for plotting

    plt.figure(figsize=(10, 6))
    plt.errorbar(performance_by_lambda['lambda'], performance_by_lambda['mean'],
                 yerr=performance_by_lambda['std'], fmt='-o', capsize=5,
                 label=f'Mean Outer Fold {metric.upper()} ± Std Dev')

    plt.xscale('log')
    plt.xlabel('Selected Lambda (log scale)')
    plt.ylabel(f'Outer Fold {metric.upper()}')
    plt.title(f'Generalization Performance vs. Selected Lambda ({metric.upper()})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()  