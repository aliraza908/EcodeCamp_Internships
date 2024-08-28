from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('best_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload and prediction
    file = request.files['file']
    img = preprocess_image(file)
    prediction = model.predict(np.expand_dims(img, axis=0))
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class = class_names[np.argmax(prediction)]
    return render_template('result.html', prediction=predicted_class)

def preprocess_image(file):
    # Load image
    img = Image.open(file.stream)
    # Resize the image to the size expected by your model
    img = img.resize((128, 128))  # Adjust the size as per your model's requirement
    # Convert the image to an array
    img_array = np.array(img)
    # Normalize the image
    img_array = img_array / 255.0
    # If your model expects a specific number of channels (e.g., 3 for RGB), ensure the image has those channels
    if img_array.ndim == 2:  # If grayscale
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
    elif img_array.shape[2] == 4:  # If RGBA
        img_array = img_array[:, :, :3]  # Convert to RGB
    return img_array

if __name__ == '__main__':
    app.run(debug=True)
