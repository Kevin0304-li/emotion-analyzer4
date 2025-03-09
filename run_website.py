import os
import sys
import subprocess

def check_requirements():
    """Check if all required libraries are installed"""
    try:
        import flask
        import torch
        import transformers
        import pandas
        import numpy
        import matplotlib
        return True
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install all requirements with: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the model output exists"""
    if not os.path.exists('model_output'):
        print("Model output directory not found.")
        print("Options:")
        print("1. Create a mock model (fast, rule-based)")
        print("2. Create a simplified demo model (slower, uses BERT)")
        print("3. Wait for full model training to complete")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nCreating a mock model...")
            import mock_model
            mock_model.create_mock_model_artifacts()
        elif choice == '2':
            print("\nCreating a simplified demo model (this may take a few minutes)...")
            import demo_model
            demo_model.create_demo_model()
        else:
            print("\nWaiting for the full model to be trained.")
            print("The website will run, but analysis features may not work until training is complete.")
        
        return False
    return True

def create_avatar():
    """Create the user avatar if it doesn't exist"""
    avatar_path = os.path.join('static', 'img', 'user-avatar.png')
    if not os.path.exists(avatar_path):
        try:
            # Try to create the avatar using PIL
            avatar_script = os.path.join('static', 'img', 'create_avatar.py')
            if os.path.exists(avatar_script):
                print("Creating user avatar...")
                subprocess.run([sys.executable, avatar_script], check=True)
            else:
                print("Warning: Avatar creation script not found.")
        except Exception as e:
            print(f"Warning: Could not create avatar image: {e}")
            print("The website will still work, but user avatars may not display correctly.")

def run():
    """Run the Flask application"""
    if not check_requirements():
        return
        
    check_model()
    create_avatar()
    
    print("\n" + "="*50)
    print("Starting SentiMind Sentiment Analysis Website")
    print("="*50)
    print("\nOpen your browser and navigate to: http://localhost:5000\n")
    
    # Import and run the Flask app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    run() 