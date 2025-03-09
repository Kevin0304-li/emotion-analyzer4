import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd

# Define a simplified version of the model for demonstration
class SimplifiedContextAwareSentimentModel(nn.Module):
    """
    A simplified version of the context-aware sentiment model for demonstration
    """
    def __init__(self, num_classes):
        super(SimplifiedContextAwareSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # Combined layers for all three components
        self.combined_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        features = torch.relu(self.combined_layer(cls_output))
        logits = self.classifier(features)
        
        return logits

def create_demo_model():
    """
    Create and save a simplified demo model
    """
    # Define emotion categories
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
    
    # Create model_output directory if it doesn't exist
    os.makedirs('model_output', exist_ok=True)
    
    # Create emotion mappings CSV
    df = pd.DataFrame({
        'emotion': emotions,
        'index': list(range(len(emotions)))
    })
    df.to_csv('model_output/emotion_mappings.csv', index=False)
    
    # Create a simplified model
    print("Creating a simplified demo model. This may take a moment...")
    model = SimplifiedContextAwareSentimentModel(num_classes=len(emotions))
    
    # Save the model state dict
    torch.save(model.state_dict(), 'model_output/model.pth')
    
    # Save the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained('model_output')
    
    print("Demo model created successfully!")
    return model, tokenizer

if __name__ == "__main__":
    create_demo_model() 