#!/usr/bin/env python3
"""
MNIST Web App Launcher

Simple launcher script for the MNIST digit recognition web application.
This script starts the Flask web server and provides helpful instructions.

Usage:
    python run_web_app.py

Author: AI Assistant
Date: 2024
"""

import sys
import os

def main():
    """
    Launch the MNIST web application.
    
    This function starts the Flask web server for the MNIST digit recognition
    web application. It provides helpful information about the server status
    and handles common errors gracefully.
    
    The web application features:
    - Interactive HTML5 canvas for drawing digits
    - Real-time AI prediction with confidence scores
    - Top 3 predictions display
    - Mobile-friendly responsive design
    - Automatic model training if no saved model exists
    
    Server runs on http://localhost:5001 by default.
    """
    print("MNIST Digit Recognition Web App")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    # Virtual environments help isolate dependencies and avoid conflicts
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment detected")
    else:
        print("Warning: Not in a virtual environment")
        print("   Consider running: source venv/bin/activate")
    
    print("\nStarting web application...")
    print("   Open your browser and go to: http://localhost:5001")
    print("   Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Launch the Flask app with error handling
    try:
        from app import app
        # Run Flask app in debug mode for development
        # host='0.0.0.0' allows access from other devices on the network
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\nWeb app stopped")
    except Exception as e:
        print(f"Error starting web app: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
