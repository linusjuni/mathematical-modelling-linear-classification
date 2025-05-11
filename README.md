# Pneumonia Detection from Chest X-rays  
**Mathematical Modeling Course - Exam Project**  

## Overview  
This project applies linear classification techniques to detect pneumonia in chest X-ray images using the MedMNIST dataset. The goal is to explore how different feature extraction methods affect model performance, especially under small dataset constraints.

## Tasks  
- Build and improve linear models for binary classification  
- Apply regularization to prevent overfitting  
- Extract features from raw pixels, image gradients, and histogram-based patches  
- Evaluate model accuracy and interpret learned weights  
- Compare feature strategies based on performance and interpretability

## How You Can Experiement With The Project
### Training Models

You can train models by running the respective `part_X.py` scripts from the `src/` directory. These scripts will save the trained models to the `models/` directory and print evaluation metrics.

1.  **Part 1: Raw Pixel Data**
    *   Uses flattened raw pixel values as features.
    *   To run: `python src/part_1.py`
    *   To modify parameters (e.g., PCA, visualization), edit the `if __name__ == "__main__":` block in `src/part_1.py`.

2.  **Part 2: Sobel Filter Features**
    *   Applies Sobel filters to extract gradient information as features.
    *   To run: `python src/part_2.py`
    *   Modify parameters similarly in the `if __name__ == "__main__":` block of `src/part_2.py`.

3.  **Part 3: Histogram of Oriented Gradients (HOG) Features**
    *   Extracts HOG features.
    *   To run: `python src/part_3.py`
    *   HOG parameters (`pixels_per_cell`, `orientations`, `cells_per_block`) are passed to `load_data_with_histograms_of_orientation` within the `run_part3` function in `src/part_3.py`. You can change them there for a specific run.
    *   Modify other parameters (PCA, visualization) in the `if __name__ == "__main__":` block of `src/part_3.py`.

**Output of Training:**
*   Trained models are saved as `.joblib` files in the `models/` directory, named with a timestamp.
*   Cross-validation results and final test set performance (Accuracy, AUC) are printed to the console.
*   If `visualize=True` is set in the script's main execution block, plots related to lambda selection, performance, and model weights will be displayed.

### Loading and Using Pre-trained Models

You can load previously trained models for evaluation or making predictions on new data.

1.  **Locate the model:** Models are saved in the `models/` directory.
2.  **Load the model:** Use the `load_model` function from `src/model_utils.py`.

    You will need to:
    *   Specify the model filename and path.
    *   Ensure any new data is preprocessed *exactly* as the original training data. This includes feature extraction (e.g., HOG with identical parameters) and PCA (using the *same fitted* PCA transformer if PCA was applied during training).
