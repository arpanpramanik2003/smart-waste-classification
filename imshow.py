import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Loading the trained model
model = tf.keras.models.load_model('best_model.keras')

# Defining the target image size
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Creating  a reverse mapping as follows:
class_indices = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
# Reversing the mapping to get index->label
idx2label = {v: k for k, v in class_indices.items()}

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load image with target size
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Convert image to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (using MobileNetV2 preprocessing)
    img_array = preprocess_input(img_array)
    return img_array, img

# Path to real-life test image
img_path = 'download.jpeg'  # Change this to your image path

# Load and preprocess the image
img_array, original_img = load_and_preprocess_image(img_path)

# Predict with the model
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = idx2label[predicted_class]
confidence = predictions[0][predicted_class]

# Display the image and the prediction result
plt.imshow(original_img)
plt.title(f"Prediction: {predicted_label} ({confidence*100:.2f}%)")
plt.axis('off')
plt.show()

print(f"Predicted Class: {predicted_label} with confidence {confidence:.4f}")
