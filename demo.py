"""
Demo script for MNIST Digit Recognition

This script demonstrates how to use the MNISTDigitRecognizer class
with a quick training session and interactive predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import MNISTDigitRecognizer

def quick_demo():
    """Run a quick demo with fewer epochs for demonstration"""
    print("=== MNIST Digit Recognition Demo ===\n")
    
    # Initialize recognizer
    recognizer = MNISTDigitRecognizer()
    
    # Load data
    print("Loading MNIST dataset...")
    recognizer.load_data()
    
    # Build model
    print("\nBuilding model...")
    recognizer.build_model()
    
    # Quick training (fewer epochs for demo)
    print("\nTraining model (5 epochs for demo)...")
    recognizer.train_model(epochs=5, batch_size=256)
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy, loss, y_pred, y_true = recognizer.evaluate_model()
    
    # Show some predictions
    print("\nShowing sample predictions...")
    recognizer.visualize_predictions(num_samples=8)
    
    return recognizer, accuracy

def interactive_prediction(recognizer):
    """Interactive prediction on random test samples"""
    print("\n=== Interactive Prediction Demo ===")
    print("Showing random test samples with predictions...")
    
    # Get random samples
    num_samples = 5
    indices = np.random.choice(len(recognizer.x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get image and true label
        image = recognizer.x_test[idx]
        true_label = np.argmax(recognizer.y_test[idx])
        
        # Make prediction
        pred_digit, confidence = recognizer.predict_digit(image)
        
        # Display
        axes[i].imshow(image.reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_digit}\nConf: {confidence:.2f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Showed {num_samples} random test samples with predictions.")

if __name__ == "__main__":
    # Run quick demo
    recognizer, accuracy = quick_demo()
    
    # Interactive prediction
    interactive_prediction(recognizer)
    
    print(f"\nDemo completed! Final accuracy: {accuracy:.4f}")
    print("You can now use the recognizer for your own predictions!")
