#!/usr/bin/env python3
"""
MNIST Desktop Application

A reliable desktop application for MNIST digit recognition using tkinter.
This version works properly on macOS by using manual canvas capture instead
of the problematic postscript method.

Features:
- Interactive drawing canvas for digit input
- Real-time AI prediction with confidence scores
- Top 3 predictions display
- Clean, professional interface
- Cross-platform compatibility

Author: AI Assistant
Date: 2024
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras

class MNISTDesktopApp:
    """
    Desktop application for MNIST digit recognition.
    
    This class provides a complete desktop interface for drawing digits
    and getting AI predictions using a trained CNN model.
    
    Attributes:
        root: Main tkinter window
        model: Trained Keras model for digit recognition
        drawing: Boolean flag for drawing state
        last_x, last_y: Last mouse position for drawing
    """
    
    def __init__(self, root):
        """
        Initialize the MNIST desktop application.
        
        Args:
            root: Main tkinter window
        """
        self.root = root
        self.root.title("MNIST Digit Recognition")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Drawing state variables
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        
        # Initialize model and GUI
        self.model = None
        self.load_model()
        self.create_widgets()
        
    def load_model(self):
        """
        Load the trained MNIST model from disk.
        
        If no saved model exists, trains a new model using the
        MNISTDigitRecognizer class and saves it for future use.
        """
        try:
            self.model = keras.models.load_model('mnist_model.h5')
            print("Model loaded successfully")
        except FileNotFoundError:
            print("No saved model found, training a new model...")
            self._train_new_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new MNIST model and save it."""
        try:
            from improved_model import ImprovedMNISTRecognizer
            
            recognizer = ImprovedMNISTRecognizer()
            recognizer.load_data()
            recognizer.build_improved_model()
            recognizer.train_improved_model(epochs=3, batch_size=512)
            
            self.model = recognizer.model
            self.model.save('mnist_model.h5')
            print("Model trained and saved successfully")
        except Exception as e:
            print(f"Error training model: {e}")
            messagebox.showerror("Error", f"Failed to train model: {e}")
    
    def create_widgets(self):
        """
        Create and layout all GUI widgets.
        
        This method sets up the main interface including the drawing canvas,
        control buttons, and prediction display area.
        """
        # Title
        title_label = tk.Label(
            self.root, 
            text="MNIST Digit Recognition", 
            font=('Arial', 24, 'bold'),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left panel - Drawing area
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Drawing canvas
        self.canvas = tk.Canvas(
            left_panel, 
            width=280, 
            height=280, 
            bg='white', 
            relief='sunken', 
            bd=2
        )
        self.canvas.pack(pady=20)
        
        # Bind drawing events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Instructions
        instructions = tk.Label(
            left_panel,
            text="Draw a digit (0-9) in the white box above",
            font=('Arial', 12),
            bg='white',
            fg='#666'
        )
        instructions.pack(pady=(0, 20))
        
        # Control buttons
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(pady=(0, 20))
        
        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_digit,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2
        )
        predict_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_canvas,
            font=('Arial', 12, 'bold'),
            bg='#f44336',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=2
        )
        clear_btn.pack(side='left', padx=5)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#f8f9fa', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Results title
        results_title = tk.Label(
            right_panel,
            text="AI Prediction",
            font=('Arial', 16, 'bold'),
            bg='#f8f9fa',
            fg='#333'
        )
        results_title.pack(pady=20)
        
        # Prediction result
        self.prediction_frame = tk.Frame(right_panel, bg='white', relief='sunken', bd=2)
        self.prediction_frame.pack(fill='x', padx=20, pady=10)
        
        # Predicted digit
        self.predicted_digit = tk.Label(
            self.prediction_frame,
            text="-",
            font=('Arial', 48, 'bold'),
            bg='white',
            fg='#333'
        )
        self.predicted_digit.pack(pady=20)
        
        # Confidence
        self.confidence_label = tk.Label(
            self.prediction_frame,
            text="Confidence: 0%",
            font=('Arial', 14),
            bg='white',
            fg='#666'
        )
        self.confidence_label.pack()
        
        # Confidence bar
        self.confidence_bar = tk.Frame(
            self.prediction_frame,
            bg='#4CAF50',
            height=20,
            width=0
        )
        self.confidence_bar.pack(pady=10, padx=20, fill='x')
        
        # Top predictions
        top_predictions_title = tk.Label(
            right_panel,
            text="Top 3 Predictions:",
            font=('Arial', 12, 'bold'),
            bg='#f8f9fa',
            fg='#333'
        )
        top_predictions_title.pack(pady=(20, 10))
        
        self.top_predictions_list = tk.Frame(right_panel, bg='#f8f9fa')
        self.top_predictions_list.pack(fill='both', expand=True, padx=20)
        
    def start_drawing(self, event):
        """
        Start drawing on the canvas.
        
        Args:
            event: Mouse event containing x, y coordinates
        """
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """
        Draw a line on the canvas from the last position to current position.
        
        Args:
            event: Mouse event containing current x, y coordinates
        """
        if self.drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=8, fill='black', capstyle='round', smooth=True
            )
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        """
        Stop drawing on the canvas.
        
        Args:
            event: Mouse event (unused but required by tkinter)
        """
        self.drawing = False
    
    def clear_canvas(self):
        """
        Clear the drawing canvas and reset all predictions.
        
        This method removes all drawn content from the canvas and
        resets the prediction display to its initial state.
        """
        self.canvas.delete("all")
        self.canvas.configure(bg='white')
        self.reset_predictions()
    
    def reset_predictions(self):
        """
        Reset the prediction display to its initial state.
        
        Clears the predicted digit, confidence, and top predictions
        display areas.
        """
        self.predicted_digit.config(text="-")
        self.confidence_label.config(text="Confidence: 0%")
        self.confidence_bar.config(width=0)
        
        # Clear top predictions
        for widget in self.top_predictions_list.winfo_children():
            widget.destroy()
    
    def predict_digit(self):
        """
        Predict the digit drawn on the canvas using the trained model.
        
        This method captures the canvas content, preprocesses it to match
        MNIST format, and runs it through the trained neural network.
        """
        try:
            # Validate that something has been drawn
            if not self._has_drawing():
                messagebox.showwarning("Warning", "Please draw something first!")
                return
            
            # Capture and preprocess the canvas image
            img_array = self._capture_canvas()
            
            # Make prediction if model is available
            if self.model is not None:
                prediction = self.model.predict(img_array, verbose=0)
                self._update_prediction_display(prediction[0])
            else:
                messagebox.showerror("Error", "Model not loaded!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            print(f"Prediction error: {e}")
    
    def _has_drawing(self):
        """
        Check if there is any content drawn on the canvas.
        
        Returns:
            bool: True if there are drawn items on the canvas
        """
        return len(self.canvas.find_all()) > 0
    
    def _capture_canvas(self):
        """
        Capture the canvas content and preprocess it for the model.
        
        Returns:
            np.ndarray: Preprocessed image array ready for model input
        """
        # Create a PIL Image by manually capturing the canvas
        width, height = 280, 280
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Redraw all line items from canvas
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'line':
                coords = self.canvas.coords(item)
                if len(coords) >= 4:
                    draw.line(coords, fill='black', width=8)
        
        # Convert to grayscale and resize to 28x28 (MNIST format)
        img = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and preprocess
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors (MNIST has white background)
        img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
        
        return img_array
    
    def _update_prediction_display(self, prediction):
        """
        Update the GUI with prediction results.
        
        Args:
            prediction: Model prediction array
        """
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Update main prediction display
        self.predicted_digit.config(text=str(predicted_digit))
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
        
        # Update confidence bar
        bar_width = int(confidence * 200)  # Max width of 200 pixels
        self.confidence_bar.config(width=bar_width)
        
        # Show top 3 predictions
        self.show_top_predictions(prediction)
    
    def show_top_predictions(self, prediction):
        """
        Display the top 3 predictions with their confidence scores.
        
        Args:
            prediction: Model prediction array
        """
        # Clear existing predictions
        for widget in self.top_predictions_list.winfo_children():
            widget.destroy()
        
        # Get top 3 predictions (sorted by confidence)
        top_indices = np.argsort(prediction)[-3:][::-1]
        
        # Create labels for each prediction
        for i, idx in enumerate(top_indices):
            conf = prediction[idx] * 100
            label = tk.Label(
                self.top_predictions_list,
                text=f"{i+1}. Digit {idx}: {conf:.1f}%",
                font=('Arial', 10),
                bg='#f8f9fa',
                fg='#333'
            )
            label.pack(anchor='w', pady=2)


def main():
    """
    Main function to run the MNIST desktop application.
    
    Creates the main tkinter window and starts the application.
    """
    root = tk.Tk()
    app = MNISTDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
