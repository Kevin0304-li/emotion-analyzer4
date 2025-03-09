import os
import argparse
import pandas as pd
import torch
from torch import nn
import numpy as np
from transformers import BertTokenizer
from sentiment_analysis_model import ContextAwareSentimentModel

def load_model_artifacts(model_dir):
    """
    Load the trained model and related artifacts
    """
    # Load emotion mappings
    emotion_mappings = pd.read_csv(os.path.join(model_dir, 'emotion_mappings.csv'))
    emotion_to_idx = {row['emotion']: row['index'] for _, row in emotion_mappings.iterrows()}
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    # Initialize model
    model = ContextAwareSentimentModel(num_classes=len(emotion_to_idx))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), 
                                     map_location=torch.device('cpu')))
    model.eval()
    
    return model, tokenizer, emotion_to_idx, idx_to_emotion

def analyze_text(text, model, tokenizer, idx_to_emotion, max_len=128, device=None):
    """
    Analyze the sentiment/emotion of a given text
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
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

def format_result(result):
    """
    Format the analysis result for display
    """
    # Sort emotions by confidence score
    sorted_emotions = sorted(result['emotion_scores'].items(), key=lambda x: x[1], reverse=True)
    
    output = [
        f"Text: {result['text']}",
        f"Predicted Emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.4f})",
        "\nEmotion Scores:"
    ]
    
    for emotion, score in sorted_emotions:
        output.append(f"  {emotion}: {score:.4f}")
    
    return "\n".join(output)

def interactive_mode(model, tokenizer, idx_to_emotion):
    """
    Run an interactive session where the user can input text for analysis
    """
    print("=== Interactive Sentiment Analysis ===")
    print("Enter text to analyze or 'quit' to exit")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            print("Please enter some text to analyze.")
            continue
        
        result = analyze_text(text, model, tokenizer, idx_to_emotion)
        print("\nAnalysis Result:")
        print(format_result(result))

def analyze_from_file(input_file, model, tokenizer, idx_to_emotion, output_file=None):
    """
    Analyze texts from a file and optionally write results to an output file
    """
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    results = []
    for text in texts:
        result = analyze_text(text, model, tokenizer, idx_to_emotion)
        results.append(result)
        print(format_result(result))
        print("-" * 50)
    
    # Write results to output file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(format_result(result))
                    f.write("\n" + "-" * 50 + "\n")
            print(f"Results written to {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Context-Aware Sentiment Analysis')
    parser.add_argument('--model_dir', type=str, default='model_output',
                        help='Directory containing the trained model and artifacts')
    parser.add_argument('--input_file', type=str,
                        help='File containing texts to analyze (one per line)')
    parser.add_argument('--output_file', type=str,
                        help='File to write analysis results to')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model and artifacts
    print(f"Loading model from {args.model_dir}...")
    model, tokenizer, emotion_to_idx, idx_to_emotion = load_model_artifacts(args.model_dir)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(model, tokenizer, idx_to_emotion)
    elif args.input_file:
        analyze_from_file(args.input_file, model, tokenizer, idx_to_emotion, args.output_file)
    else:
        print("No input provided. Use --interactive or --input_file.")
        parser.print_help()

if __name__ == "__main__":
    main() 