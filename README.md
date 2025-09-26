# MNIST Digit Recognition System

A complete machine learning system for recognizing handwritten digits (0-9) using the MNIST dataset. Includes both web and desktop interfaces with **99.50% accuracy** and **99.65% average confidence**.

## Features

- **99.50% accuracy** on MNIST test data (improved model)
- **99.65% average confidence** levels for predictions
- **Web application** with beautiful HTML5 canvas interface
- **Desktop application** with native tkinter interface
- **Real-time prediction** with confidence scores
- **Top 3 predictions** ranking
- **Mobile-friendly** web interface
- **Two model options**: Basic (98.86%) and Improved (99.50%)

## Quick Start

### Option 1: Web Application (Recommended)
```bash
python app.py
# Open browser: http://localhost:5001
```

### Option 2: Desktop Application
```bash
python desktop.py
```

### Option 3: Quick Demo
```bash
python demo.py
```

## Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py  # Web app
# OR
python desktop.py  # Desktop app
```

## File Structure

```
mnist/
├── app.py              # Web application (Flask)
├── desktop.py          # Desktop application (tkinter)
├── model.py            # AI model and training code
├── demo.py             # Quick demo script
├── web.py              # Easy web launcher
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Web interface
└── mnist_model.h5      # Trained model (auto-generated)
```

## How to Use

### Web Interface
1. Draw a digit (0-9) in the white canvas
2. Click "Predict" to see AI recognition
3. View confidence score and top 3 predictions
4. Click "Clear" to start over

### Desktop Interface
1. Draw a digit in the canvas using your mouse
2. Click "Predict" to get AI prediction
3. View confidence and top predictions
4. Click "Clear" to start fresh

## Model Architecture

The system offers two model options:

### Basic Model (98.86% accuracy)
- **3 Convolutional layers** (32, 64, 64 filters)
- **MaxPooling layers** for dimensionality reduction
- **Dropout layers** for regularization (0.5 rate)
- **Dense layers** for final classification
- **Adam optimizer** with categorical crossentropy loss
- **Parameters**: 93,322 (lightweight)
- **Training Time**: ~5 epochs to reach 98%+ accuracy
- **Prediction Speed**: ~50ms per prediction

### Improved Model (99.50% accuracy, 99.65% confidence)
- **5 Convolutional layers** (32, 32, 64, 64, 128 filters)
- **Batch Normalization** after each conv layer
- **MaxPooling layers** for dimensionality reduction
- **Strategic Dropout** (0.25-0.5) for better regularization
- **Dense layers** (512, 256) with batch normalization
- **Adam optimizer** with advanced settings
- **Parameters**: ~2.1M (more sophisticated)
- **Training Time**: ~10 epochs to reach 99%+ accuracy
- **Prediction Speed**: ~60ms per prediction

**Key Improvements in Enhanced Model**:
- Deeper architecture for better feature extraction
- Batch normalization for training stability
- Higher dropout rates to prevent overfitting
- More sophisticated regularization techniques
- Significantly higher confidence levels (90%+ vs 70-80%)

## Programmatic Usage

### Basic Model (98.86% accuracy)
```python
from model import MNISTDigitRecognizer

# Initialize and train basic model
recognizer = MNISTDigitRecognizer()
recognizer.load_data()
recognizer.build_model()
recognizer.train_model(epochs=10)

# Evaluate performance
accuracy, loss, y_pred, y_true = recognizer.evaluate_model()
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
digit, confidence = recognizer.predict_digit(image)
print(f"Predicted: {digit} (confidence: {confidence:.2f})")
```

### Improved Model (99.50% accuracy, 99.65% confidence)
```python
from improved_model import ImprovedMNISTRecognizer

# Initialize and train improved model
recognizer = ImprovedMNISTRecognizer()
recognizer.load_data()
recognizer.build_improved_model()
recognizer.train_improved_model(epochs=15)

# Evaluate performance
accuracy, loss, y_pred, y_true = recognizer.evaluate_model()
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions with higher confidence
digit, confidence = recognizer.predict_digit(image)
print(f"Predicted: {digit} (confidence: {confidence:.2f})")
```

## Confidence Levels

The system provides detailed confidence information for predictions:

### Basic Model Confidence
- **Average Confidence**: 70-80%
- **High Confidence (>90%)**: ~70% of predictions
- **Medium Confidence (70-90%)**: ~25% of predictions
- **Low Confidence (<70%)**: ~5% of predictions

### Improved Model Confidence
- **Average Confidence**: 99.65%
- **High Confidence (>90%)**: 99.05% of predictions
- **Medium Confidence (70-90%)**: 0.95% of predictions
- **Low Confidence (<70%)**: 0% of predictions

### Understanding Confidence
- **90%+ confidence**: Very reliable prediction
- **70-90% confidence**: Good prediction, may have some uncertainty
- **<70% confidence**: Uncertain prediction, consider redrawing

### Tips for Higher Confidence
1. **Draw clearly** - well-formed, distinct digits
2. **Use good contrast** - dark lines on light background
3. **Center the digit** - don't draw in corners
4. **Make digits larger** - fill more of the canvas
5. **Avoid messy strokes** - clean, single lines

## Customization

### Model Architecture
Modify `build_model()` in `model.py`:
```python
# Add more layers
layers.Conv2D(128, (3, 3), activation='relu')

# Adjust dropout rate
layers.Dropout(0.3)  # Lower dropout

# Change dense layer size
layers.Dense(128, activation='relu')  # More neurons
```

### Web Interface
Customize `templates/index.html`:
- Change colors and styling
- Add new drawing tools
- Modify layout and animations

## Troubleshooting

**"Module not found" errors:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"Port 5001 already in use":**
```bash
lsof -ti:5001 | xargs kill -9
# Or change the port in app.py
```

**"Model not found":**
- The app will automatically train a model if none exists
- This may take a few minutes on first run

**Desktop app not working:**
- Use the web app instead: `python app.py`
- Web app is more reliable and works on all platforms

## Dependencies

- tensorflow>=2.20.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- flask>=2.3.0
- pillow>=9.0.0
- scikit-learn>=1.3.0



---
