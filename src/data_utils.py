import numpy as np
import os
from PIL import Image
import glob
from scipy import ndimage
from skimage.feature import hog


# Linus: /Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data
# Ask: 
# Simon:

def load_data(folder_path):
    X = []
    y = []
    
    image_files = glob.glob(os.path.join(folder_path, "*.png"), recursive=True)
    
    print(f"Found {len(image_files)} PNG images in {folder_path}")
    
    for file_path in image_files:
        filename = os.path.basename(file_path).lower()
        if "positive" in filename:
            label = 1  # Pneumonia
        else:
            label = 0  # Healthy
            
        try:
            img = Image.open(file_path).convert('L')
            img_array = np.array(img).flatten() / 255.0
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {len(X)} images: {sum(y)} pneumonia, {len(X) - sum(y)} healthy")
    
    return np.array(X), np.array(y)

def load_data_with_sobel_kernel(folder_path):
    X_raw, y = load_data(folder_path)

    X_gradient = []

    for img_flat in X_raw:
        img_2d = img_flat.reshape(224, 224)

        gradient_x = ndimage.sobel(img_2d, axis=1)
        gradient_y = ndimage.sobel(img_2d, axis=0)

        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        X_gradient.append(gradient_magnitude.flatten())

    print(f"Processed {len(X_gradient)} images with Sobel gradient features")
    
    return np.array(X_gradient), y

def load_data_with_histograms_of_orientation(folder_path, pixels_per_cell=(32, 32), orientations=72, cells_per_block=(3, 3)):
    X_raw, y = load_data(folder_path)

    X_features = []

    for img_flat in X_raw:
        img_2d = img_flat.reshape(224, 224)

        hog_features = hog(
            img_2d,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        X_features.append(hog_features)    

    print(f"Processed {len(X_features)} images with HOG features")
    print(f"First 3 vectors in X_features size: {[len(X_features[i]) for i in range(min(3, len(X_features)))]}")
    
    return np.array(X_features), y



