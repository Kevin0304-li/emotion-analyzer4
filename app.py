from flask import Flask, render_template, request, jsonify
import torch
import os
import sys
import json
import importlib.util

# Add the current directory to the path to ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the models, starting with the real model
model_type = "unknown"
try:
    from sentiment_analysis_model import ContextAwareSentimentModel
    model_class = ContextAwareSentimentModel
    model_type = "real"
    print("Using real context-aware sentiment model.")
except ImportError:
    try:
        from demo_model import SimplifiedContextAwareSentimentModel
        model_class = SimplifiedContextAwareSentimentModel
        model_type = "demo"
        print("Using simplified demo model.")
    except ImportError:
        model_type = "mock"
        print("Using mock model.")

# Import the mock model for fallback
import mock_model

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables to store model, tokenizer, and emotion mapping
model = None
tokenizer = None
idx_to_emotion = None
emotion_to_idx = None
using_mock = False

def load_model_artifacts(model_dir='model_output'):
    """
    Load the trained model and related artifacts
    """
    global model, tokenizer, idx_to_emotion, emotion_to_idx, using_mock, model_type
    
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' not found. Creating mock model.")
        mock_model.create_mock_model_artifacts()
        model_type = "mock"
    
    try:
        # Load emotion mappings
        with open(os.path.join(model_dir, 'emotion_mappings.csv'), 'r') as f:
            # Skip header
            next(f)
            # Parse CSV data
            emotion_mappings = {}
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        emotion, idx = parts[0], int(parts[1])
                        emotion_mappings[emotion] = idx
        
        emotion_to_idx = emotion_mappings
        idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
        
        # Load tokenizer
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # Try to load the appropriate model
        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        if model_type == "real":
            from sentiment_analysis_model import ContextAwareSentimentModel
            model = ContextAwareSentimentModel(num_classes=len(emotion_to_idx))
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            using_mock = False
            print("Loaded real sentiment analysis model.")
        elif model_type == "demo":
            from demo_model import SimplifiedContextAwareSentimentModel
            model = SimplifiedContextAwareSentimentModel(num_classes=len(emotion_to_idx))
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            using_mock = False
            print("Loaded simplified demo model.")
        else:
            model = mock_model.MockSentimentModel(num_classes=len(emotion_to_idx))
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            using_mock = True
            print("Loaded mock sentiment analysis model.")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to mock model.")
        mock_model.create_mock_model_artifacts()
        model_type = "mock"
        using_mock = True
        return load_model_artifacts()  # Try again with the mock model

def analyze_text(text, max_len=128):
    """
    Analyze the sentiment/emotion of a given text
    """
    global model, tokenizer, idx_to_emotion, using_mock
    
    # Check if model is loaded
    if model is None or tokenizer is None or idx_to_emotion is None:
        if not load_model_artifacts():
            return {"error": "Model not loaded properly"}
    
    # If using mock model, use the mock analysis
    if using_mock:
        return mock_model.analyze_with_mock_model(text)
    
    # Otherwise use the real or demo model
    device = torch.device("cpu")  # Use CPU for web deployment
    
    # Tokenize text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(probabilities, dim=1)
        predicted_emotion = idx_to_emotion[predicted_class.item()]
    
    # Get confidence scores for all emotions
    confidence_scores = {idx_to_emotion[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    result = {
        'text': text,
        'predicted_emotion': predicted_emotion,
        'confidence': probabilities[0][predicted_class].item(),
        'emotion_scores': confidence_scores
    }
    
    return result

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for analyzing text"""
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    # Analyze the text
    result = analyze_text(data['text'])
    
    return jsonify(result)

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/model-status')
def model_status():
    """Return the status of the model (real, demo, or mock)"""
    global using_mock, model_type
    
    # Load model if not loaded
    if model is None:
        load_model_artifacts()
    
    model_info = {
        "real": "Real Context-Aware Sentiment Model",
        "demo": "Simplified Demo Model",
        "mock": "Mock Rule-Based Model"
    }
        
    return jsonify({
        "using_mock": using_mock,
        "model_type": model_type,
        "model_description": model_info.get(model_type, "Unknown Model Type")
    })

if __name__ == '__main__':
    # Try to load model at startup
    load_model_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000) 