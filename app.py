"""
MNIST Digit Recognition Web Application

A Flask-based web application that provides an interactive interface for MNIST digit recognition.
Users can draw digits on a canvas and receive real-time AI predictions with confidence scores.

Features:
- Interactive HTML5 canvas for drawing digits
- Real-time prediction using trained CNN model
- Confidence visualization and top-3 predictions
- Responsive design for desktop and mobile devices
- Automatic model training if no saved model exists

Author: AI Assistant
Date: 2024
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Initialize Flask application
app = Flask(__name__)

# Global variable to store the trained model
model = None

def load_model():
    """
    Load the trained MNIST model for digit recognition.
    
    This function attempts to load a pre-trained model from disk. If no saved model
    exists, it will train a new model using the MNISTDigitRecognizer class.
    
    The model is stored globally to avoid reloading on each prediction request.
    """
    global model
    try:
        # Try to load the saved improved model first
        model = keras.models.load_model('mnist_model.h5')
        print("Loaded saved improved model")
    except:
        # If no saved model exists, train a new improved model
        print("No saved model found, training improved model...")
        from improved_model import ImprovedMNISTRecognizer
        
        # Initialize and train the improved model
        recognizer = ImprovedMNISTRecognizer()
        recognizer.load_data()
        recognizer.build_improved_model()
        recognizer.train_improved_model(epochs=5, batch_size=128)  # Quick training for demo
        
        # Store the trained model globally and save it
        model = recognizer.model
        model.save('mnist_model.h5')
        print("Improved model trained and saved")

def preprocess_image(image_data):
    """Preprocess the drawn image for prediction"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST size)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert colors (MNIST has white background, black digits)
        img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Main page with drawing interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from drawn image"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        if model is None:
            load_model()
        
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get top 3 predictions
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        top3_predictions = [
            {'digit': int(idx), 'confidence': float(prediction[0][idx])}
            for idx in top3_indices
        ]
        
        return jsonify({
            'predicted_digit': int(predicted_digit),
            'confidence': confidence,
            'top3_predictions': top3_predictions
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear the canvas"""
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    print("Starting MNIST Digit Recognition Web App...")
    load_model()
    print("Web app ready! Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
