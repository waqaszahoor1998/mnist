#!/usr/bin/env python3
"""
MNIST Digit Recognition Setup Script

This script helps set up the MNIST digit recognition project by:
1. Creating a virtual environment
2. Installing dependencies
3. Training the improved model
4. Providing usage instructions

Usage:
    python setup.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("MNIST Digit Recognition Setup")
    print("=" * 50)
    
    # Check if Python 3 is available
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python {sys.version.split()[0]} detected")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Train the improved model
    print("\n🤖 Training the improved model...")
    print("This may take a few minutes...")
    
    if not run_command(f"{activate_cmd} && python improved_model.py", "Training improved model"):
        print(" Model training failed, but you can still use the basic model")
    
    print("\n Setup completed successfully!")
    print("\n Next steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run the applications:")
    print("   Web app:     python web.py")
    print("   Desktop app: python desktop.py")
    print("   Demo:        python demo.py")
    
    print("\n3. Open your browser to http://localhost:5001 for the web app")
    print("\n See README.md for detailed documentation")

if __name__ == "__main__":
    main()
