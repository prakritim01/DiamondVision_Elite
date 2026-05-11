import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_diamond_image(image_path):
    # Resize to MobileNetV2 standards
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Add logic here to load your shape_model.h5 later