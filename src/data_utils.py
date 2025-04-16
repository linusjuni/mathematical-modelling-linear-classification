import numpy as np
import os
from PIL import Image
import glob

def load_data(folder_path):
    # Linus: /Users/linus.juni/Documents/Personal/mathematical-modelling-linear-classification/data
    # Ask: 
    # Simon:
    
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