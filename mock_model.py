import os
import random
import json
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer

# Define a simple mock model for demonstration
class MockSentimentModel(nn.Module):
    def __init__(self, num_classes=8):
        super(MockSentimentModel, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask):
        # Create a random output tensor for demonstration
        batch_size = input_ids.size(0)
        return torch.randn(batch_size, self.num_classes)

def create_mock_model_artifacts():
    """
    Create mock model artifacts for demonstration purposes
    """
    print("Creating mock model artifacts for demonstration...")
    
    # Create model_output directory if it doesn't exist
    os.makedirs('model_output', exist_ok=True)
    
    # Define emotion categories
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
    
    # Create emotion mappings CSV
    df = pd.DataFrame({
        'emotion': emotions,
        'index': list(range(len(emotions)))
    })
    df.to_csv('model_output/emotion_mappings.csv', index=False)
    
    # Save a mock model state dict
    model = MockSentimentModel(num_classes=len(emotions))
    torch.save(model.state_dict(), 'model_output/model.pth')
    
    # Copy BERT tokenizer files (use the real tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained('model_output')
    
    print("Mock model artifacts created successfully.")

def analyze_with_mock_model(text):
    """
    Perform a mock sentiment analysis for demonstration
    """
    # Define emotion categories
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
    
    # Analyze text based on simple keywords for demonstration
    text_lower = text.lower()
    
    # Define keywords for each emotion for simple matching
    emotion_keywords = {
        'joy': ['happy', 'joy', 'glad', 'excited', 'wonderful', 'love', 'great', 'amazing'],
        'sadness': ['sad', 'upset', 'unhappy', 'depressed', 'miserable', 'crying', 'disappointed'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'kill', 'hate', 'rage'],
        'fear': ['afraid', 'scared', 'terrified', 'anxious', 'nervous', 'worry'],
        'surprise': ['surprised', 'shocked', 'astonished', 'amazed', 'unexpected'],
        'disgust': ['disgusting', 'gross', 'revolting', 'nasty', 'sick'],
        'trust': ['trust', 'reliable', 'faithful', 'believe', 'honest', 'loyal'],
        'anticipation': ['await', 'expect', 'anticipate', 'looking forward', 'exciting']
    }
    
    # Check for context modifiers (like "just kidding")
    context_modifiers = [
        'just kidding', 'kidding', 'joking', 'haha', 'jk', 'lol', 'friend'
    ]
    
    # Generate scores with some randomness but bias toward the matching keywords
    scores = {}
    has_context_modifier = any(modifier in text_lower for modifier in context_modifiers)
    
    for emotion in emotions:
        # Base score with randomness
        base_score = random.uniform(0.05, 0.15)
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in emotion_keywords[emotion] if keyword in text_lower)
        keyword_score = min(0.7, keyword_matches * 0.2)  # Cap at 0.7
        
        # Apply context modifiers
        if emotion == 'anger' and 'kill' in text_lower and has_context_modifier:
            # If contains "kill" but also has context modifiers, reduce anger and increase joy
            keyword_score = 0.1
            if emotion == 'joy':
                keyword_score = 0.6
        
        scores[emotion] = base_score + keyword_score
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: v/total for k, v in scores.items()}
    
    # Get the predicted emotion (highest score)
    predicted_emotion = max(scores, key=scores.get)
    confidence = scores[predicted_emotion]
    
    # Special case for the example
    if "i will kill you" in text_lower:
        if has_context_modifier:
            predicted_emotion = "joy"
            scores["joy"] = 0.6
            scores["anger"] = 0.1
            confidence = 0.6
        else:
            predicted_emotion = "anger"
            scores["anger"] = 0.7
            scores["fear"] = 0.15
            confidence = 0.7
    
    result = {
        'text': text,
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'emotion_scores': scores
    }
    
    return result

if __name__ == "__main__":
    create_mock_model_artifacts() 