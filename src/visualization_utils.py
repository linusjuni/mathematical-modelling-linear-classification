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

def visualize_weight_from_feature_vector(model, image_shape=(224, 224), pixels_per_cell=(32, 32), orientations=9):

    weights = model.coef_[0]

    n_cells_y = image_shape[0] // pixels_per_cell[0]
    n_cells_x = image_shape[1] // pixels_per_cell[1]
    
    expected_len = n_cells_y * n_cells_x * orientations
    if len(weights) != expected_len:
        print(f"Warning: Weight vector length ({len(weights)}) does not match "
              f"expected HOG feature length ({expected_len} = "
              f"{n_cells_y}x{n_cells_x} cells * {orientations} orientations). Cannot visualize weights accurately.")
        return

    try:
        weights_reshaped = weights.reshape((n_cells_y, n_cells_x, orientations))
    except ValueError as e:
        print(f"Error reshaping weights vector of length {len(weights)} "
              f"into ({n_cells_y}, {n_cells_x}, {orientations}). Cannot visualize. Error: {e}")
        return

    cell_importance = np.sum(np.abs(weights_reshaped), axis=2) # Shape: (n_cells_y, n_cells_x)

    upscaled_heatmap = np.kron(cell_importance, np.ones(pixels_per_cell))

    plt.figure(figsize=(10, 10))
    im = plt.imshow(upscaled_heatmap, cmap='viridis', extent=(0, image_shape[1], image_shape[0], 0))
    plt.colorbar(im, label='Aggregated Absolute Weight per Cell')
    plt.title(f'HOG Feature Importance Heatmap ({pixels_per_cell[0]}x{pixels_per_cell[1]} cells)')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Image Height (pixels)')
    
    ax = plt.gca()
    ax.set_xticks(np.arange(0, image_shape[1] + 1, pixels_per_cell[1]), minor=True)
    ax.set_yticks(np.arange(0, image_shape[0] + 1, pixels_per_cell[0]), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=1, alpha=0.5) 
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False, labeltop=False, labelright=False) 

    plt.show()