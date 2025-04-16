import matplotlib.pyplot as plt
import numpy as np

def visualize_weights(model, image_shape=(224, 224)):
    # Extract model weights (excluding bias)
    weights = model.coef_[0]
    
    weight_image = weights.reshape(image_shape)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(weight_image, cmap='seismic')
    plt.colorbar()
    plt.title('Model Weights (Blue = Negative, Red = Positive)')
    plt.show()