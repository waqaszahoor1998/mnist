"""
MNIST Digit Recognition System

A complete machine learning pipeline for recognizing handwritten digits (0-9) using the MNIST dataset.
This implementation includes data preprocessing, CNN model training, evaluation, and visualization tools.

Features:
- Automatic data loading and preprocessing
- Convolutional Neural Network architecture
- Model training with callbacks and validation
- Comprehensive evaluation with metrics and visualizations
- Model persistence (save/load functionality)
- Real-time prediction capabilities

Author: AI Assistant
Date: 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MNISTDigitRecognizer:
    """
    A complete MNIST digit recognition system using Convolutional Neural Networks.
    
    This class provides a comprehensive pipeline for:
    - Loading and preprocessing MNIST data
    - Building and training CNN models
    - Evaluating model performance
    - Making predictions on new images
    - Visualizing results and training history
    
    Attributes:
        model: Trained Keras model for digit recognition
        history: Training history from model.fit()
        x_train, y_train: Training data and labels
        x_test, y_test: Test data and labels
    """
    
    def __init__(self):
        """Initialize the MNIST digit recognizer with empty attributes."""
        self.model = None          # Trained Keras model
        self.history = None        # Training history
        self.x_train = None        # Training images
        self.y_train = None        # Training labels
        self.x_test = None         # Test images
        self.y_test = None         # Test labels
        
    def load_data(self):
        """
        Load and preprocess the MNIST dataset for training and evaluation.
        
        This method:
        1. Downloads MNIST dataset from Keras (if not already cached)
        2. Normalizes pixel values from [0, 255] to [0, 1]
        3. Reshapes images to include channel dimension for CNN
        4. Converts labels to categorical format for multi-class classification
        
        Returns:
            tuple: (x_train, y_train, x_test, y_test) - Preprocessed training and test data
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST dataset from Keras (automatically downloads if needed)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values from [0, 255] to [0, 1] for better training stability
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Reshape data for CNN: add channel dimension (28, 28) -> (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert integer labels to categorical (one-hot encoding) for 10 classes
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Store data as instance variables
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Print data shapes for verification
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return x_train, y_train, x_test, y_test
    
    def build_model(self):
        """
        Build a Convolutional Neural Network model for MNIST digit recognition.
        
        Architecture:
        - 3 Convolutional layers with increasing filters (32, 64, 64)
        - MaxPooling layers for dimensionality reduction
        - Dropout layers for regularization
        - Dense layers for final classification
        
        The model uses:
        - ReLU activation for hidden layers
        - Softmax activation for output (10 classes)
        - Adam optimizer for training
        - Categorical crossentropy loss for multi-class classification
        
        Returns:
            keras.Model: Compiled CNN model ready for training
        """
        print("Building CNN model...")
        
        model = keras.Sequential([
            # First convolutional block: 32 filters, 3x3 kernel
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),  # Reduces spatial dimensions by half
            
            # Second convolutional block: 64 filters, 3x3 kernel
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),  # Further reduces spatial dimensions
            
            # Third convolutional block: 64 filters, 3x3 kernel
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Classification head
            layers.Flatten(),                    # Convert 2D features to 1D
            layers.Dropout(0.5),                 # Regularization to prevent overfitting
            layers.Dense(64, activation='relu'), # Hidden dense layer
            layers.Dropout(0.5),                 # Additional regularization
            layers.Dense(10, activation='softmax') # Output layer for 10 digit classes
        ])
        
        # Compile model with appropriate optimizer, loss, and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate for stability
            loss='categorical_crossentropy',     # Loss function for multi-class classification
            metrics=['accuracy']                 # Track accuracy during training
        )
        
        self.model = model
        print("Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"Training model for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        return test_accuracy, test_loss, y_pred_classes, y_true_classes
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def visualize_predictions(self, num_samples=10):
        """Visualize sample predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        x_sample = self.x_test[indices]
        y_true = np.argmax(self.y_test[indices], axis=1)
        
        # Make predictions
        y_pred = self.model.predict(x_sample)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Plot samples
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(x_sample[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {y_true[i]}, Pred: {y_pred_classes[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='mnist_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='mnist_model.h5'):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_digit(self, image):
        """Predict digit from a single image"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Ensure image is in correct format
        if image.shape != (28, 28, 1):
            if image.shape == (28, 28):
                image = image.reshape(28, 28, 1)
            else:
                raise ValueError("Image must be 28x28 pixels")
        
        # Normalize and add batch dimension
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return digit, confidence

def main():
    """Main function to run the complete pipeline"""
    print("=== MNIST Digit Recognition System ===\n")
    
    # Initialize recognizer
    recognizer = MNISTDigitRecognizer()
    
    # Load data
    recognizer.load_data()
    
    # Build model
    recognizer.build_model()
    
    # Display model architecture
    print("\nModel Architecture:")
    recognizer.model.summary()
    
    # Train model
    recognizer.train_model(epochs=10)
    
    # Evaluate model
    test_accuracy, test_loss, y_pred, y_true = recognizer.evaluate_model()
    
    # Visualize results
    print("\nGenerating visualizations...")
    recognizer.plot_training_history()
    recognizer.plot_confusion_matrix(y_true, y_pred)
    recognizer.visualize_predictions()
    
    # Save model
    recognizer.save_model()
    
    print(f"\n=== Training Complete ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
